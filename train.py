import torch
import torchvision
from torchvision import transforms
from diffusers import UNet2DModel
from torch.optim import Adam
from tqdm import tqdm 

from itertools import chain

# @torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T

    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(0) > G.size(1):
        X = X.T
    return X


class NSPrecond:

    def __init__(self, **kwargs):
        self.steps = kwargs.get('steps', 10)

    def __call__(self, G):
        og_norm = G.norm()
        G = zeropower_via_newtonschulz5(G, steps=self.steps)
        G = G / G.norm()
        G = G * og_norm
        return G


class KronPrecond:

    def __init__(self, **kwargs):
        self.beta = kwargs.get('beta', 0.9)
        self.do_merge_dims = kwargs.get('do_merge_dims', False)
        self.GG = []
        self.Q = []
        self.max_precond_dim = kwargs.get('max_precond_dim', 2**16)
        self.step = 0

    def get_orthogonal_matrix(self, mat):
        """
        Computes the eigenbases of the preconditioner using torch.linalg.eigh decomposition.
        """
        matrix = []
        for m in mat:
            # -1 or -2
            if not isinstance(m, torch.Tensor):
                matrix.append(m)
                continue
            if m.data.dtype != torch.float:
                float_data = False
                original_type = m.data.dtype
                original_device = m.data.device
                matrix.append(m.data.float())
            else:
                float_data = True
                matrix.append(m.data)

        final = []
        for m in matrix:
            if not isinstance(m, torch.Tensor):
                final.append(m)
                continue
            try:
                _, Q = torch.linalg.eigh(m + 1e-30 * torch.eye(m.shape[0], device=m.device))
            except:
                eye = torch.eye(m.shape[0], device=m.device)
                _, Q = torch.linalg.eigh(m.to(torch.float64) + 1e-30 * eye)
                Q = Q.to(m.dtype)
            Q = torch.flip(Q, [1])

            if not float_data:
                Q = Q.to(original_device).type(original_type)
            final.append(Q)
        return final

    def get_orthogonal_matrix_QR(self, state, max_precond_dim=10000, merge_dims=False):
        """
        Computes the eigenbases of the preconditioner using one round of power iteration
        followed by torch.linalg.qr decomposition.
        """
        precond_list = self.GG
        orth_list = self.Q

        matrix = []
        orth_matrix = []
        for m, o in zip(precond_list, orth_list):
            if not isinstance(m, torch.Tensor):
                matrix.append(m)
                orth_matrix.append(o)
                continue
            if m.data.dtype != torch.float:
                float_data = False
                original_type = m.data.dtype
                original_device = m.data.device
                matrix.append(m.data.float())
                orth_matrix.append(o.data.float())
            else:
                float_data = True
                matrix.append(m.data.float())
                orth_matrix.append(o.data.float())

        orig_shape = state["exp_avg_sq"].shape
        # caution with these
        if self.do_merge_dims:
            exp_avg_sq = self.merge_dims(state["exp_avg_sq"], self.max_precond_dim)
        else:
            exp_avg_sq = state["exp_avg_sq"]

        final = []
        for ind, (m, o) in enumerate(zip(matrix, orth_matrix)):
            if not isinstance(m, torch.Tensor):
                final.append(m)
                continue
            est_eig = torch.diag(o.T @ m @ o)
            sort_idx = torch.argsort(est_eig, descending=True)
            exp_avg_sq = exp_avg_sq.index_select(ind, sort_idx)
            o = o[:, sort_idx]
            power_iter = m @ o
            Q, _ = torch.linalg.qr(power_iter)

            if not float_data:
                Q = Q.to(original_device).type(original_type)
            final.append(Q)

        if merge_dims:
            exp_avg_sq = exp_avg_sq.reshape(orig_shape)

        state["exp_avg_sq"] = exp_avg_sq
        return final

    def update_preconditioner(self, grad, state):
        """
        Updates the preconditioner matrices and the eigenbases (L, R, Q_L, Q_R in the paper).
        """
        gr = self.merge_dims(grad, self.max_precond_dim) if self.do_merge_dims else grad
        initialize = len(self.GG) == 0
        for idx, sh in enumerate(gr.shape):
            if sh <= self.max_precond_dim:
                outer_product = torch.tensordot(
                    gr,
                    gr,
                    # Contracts across all dimensions except for k.
                    dims=[[*chain(range(idx), range(idx + 1, len(grad.shape)))]] * 2,
                )
                if initialize:
                    self.GG[idx] = outer_product
                else:
                    self.GG[idx].lerp_(outer_product, 1 - self.beta)
                     
        if len(self.Q) == 0:
            self.Q = self.get_orthogonal_matrix(self.GG)
        if self.step > 0 and self.step % self.precondition_frequency == 0:
            self.Q = self.get_orthogonal_matrix_QR(state, self.max_precond_dim, self.do_merge_dims)  


    def project(self, grad, back=False):
        """
        Projects the gradient back to the original space.
        """
        last_dim = 1 if back else 0

        original_shape = grad.shape
        if self.do_merge_dims:
            grad = self.merge_dims(grad, self.max_precond_dim)
        for mat in self.Q:
            if isinstance(mat, torch.Tensor):
                grad = torch.tensordot(
                    grad,
                    mat,
                    dims=[[0], [last_dim]],
                )
            elif mat == -1:
                permute_order = list(range(1, len(grad.shape))) + [0]
                grad = grad.permute(permute_order)

        if self.do_merge_dims:
            grad = grad.reshape(original_shape)
        return grad

    def merge_dims(self, grad, max_precond_dim):
        """
        Merges dimensions of the gradient tensor till the product of the dimensions is less than or equal to max_precond_dim.
        """
        shape = grad.shape
        new_shape = []
        
        curr_shape = 1
        for sh in shape:
            temp_shape = curr_shape * sh
            if temp_shape > max_precond_dim:
                if curr_shape > 1:
                    new_shape.append(curr_shape)
                    curr_shape = sh
                else:
                    new_shape.append(sh)
                    curr_shape = 1
            else:
                curr_shape = temp_shape
        
        if curr_shape > 1 or len(new_shape)==0:
            new_shape.append(curr_shape)
        
        new_grad = grad.reshape(new_shape)
        return new_grad       

    def __call__(self, G):
        G = G.to(torch.bfloat16)
        G = G.to(torch.float16)
        G = G.to(torch.float32)
        return G


class HessPrecond:

    def __init__(self, **kwargs):
        self.beta = kwargs.get('beta', 0.9)
        self.ema = None

    def __call__(self, G):
        outer_product = G.flatten()[None, :] @ G.flatten()[:, None]
        if self.ema is None:
            self.ema = outer_product
        else:
            self.ema.lerp_(outer_product, 1 - self.beta)

        og_norm = G.norm()
        G = self.ema.sqrt() @ G

        return G


def get_model():
    block_out_channels=(128, 128, 256, 256, 512, 512)
    down_block_types=( 
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D", 
        "DownBlock2D", 
        "DownBlock2D", 
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    )
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D"  
    )
    return UNet2DModel(block_out_channels=block_out_channels,out_channels=3, in_channels=3, up_block_types=up_block_types, down_block_types=down_block_types, add_attention=True)


@torch.no_grad()
def gen_sample(model, x0, steps, c=None, precond_mode=None):
    xt = x0
    dt = 1/steps
    for t in range(steps):
        t = t/steps
        t = torch.tensor(t, device=x0.device)

        d = model(xt, t)['sample']
        xt = xt - (dt) * d

    return xt


datasets = dict(
    cifar10 = dict(
        class_name = torchvision.datasets.CIFAR10,
        transform = lambda img_size, rand_flip: transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(0.5 if rand_flip else 0),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        inverse_transform = lambda x: (x*2)-1
    ),
    mnist = dict(
        class_name = torchvision.datasets.MNIST,
        transform = lambda img_size, rand_flip: transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(0.5 if rand_flip else 0),
            transforms.ToTensor()
        ]),
        inverse_transform = lambda x: x
    )
)


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image_size = 64
    rand_flip = True
    batch_size = 64
    num_workers = 4
    num_epochs = 100
    lr = 1e-4
    log_normal_sample = False
    dataset_name = 'cifar10'
    precond_mode = None, # None, 'ns', 'kron', 'hessian'

    transform = datasets[dataset_name]['transform'](image_size, rand_flip)
    train_dataset = datasets[dataset_name]['class_name'](split='train', download=True, transform=transform)

    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    model = get_model().to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    nb_iter = 0
    print('Start training')
    for current_epoch in range(num_epochs):
        for i, data in tqdm(enumerate(dataloader)):
            x1 = data[0].to(device)
            x0 = torch.randn_like(x1)
            flow = x0 - x1
            bs = x0.shape[0]

            if log_normal_sample:
                t = torch.randn(bs, device=device)
                t = torch.sigmoid(t)
            else:
                t = torch.rand(bs, device=device)
            xt = t.view(-1,1,1,1) * x1 + (1-t).view(-1,1,1,1) * x0
            
            pred_flow = model(xt, t)['sample']
            loss = torch.nn.functional.mse_loss(pred_flow, flow)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            nb_iter += 1

            if nb_iter % 200 == 0:
                with torch.no_grad():
                    print(f'Save export {nb_iter}')
                    sample = (gen_sample(model, x0, steps=128) * 0.5) + 0.5
                    torchvision.utils.save_image(sample, f'export_{str(nb_iter).zfill(8)}.png')
                    torch.save(model.state_dict(), f'{dataset_name}.ckpt')