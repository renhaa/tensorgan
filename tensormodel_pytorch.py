import torch 
import tntorch as tn
from tqdm import tqdm
from matplotlib import pyplot as plt

class TensorModel:
    def __init__(self, T, ranks_tucker = None): 

        ## Preprocessing (subtract mean shape)
        self.T = T if type(T) == torch.Tensor else torch.tensor(T)
        self.order = len(self.T.shape)
        self.mean_shape = self.T.mean(tuple([i for i in range(1,self.order)]))
        self.mean_tensor = torch.einsum(','.join(self.get_chars(self.order)),*[self.mean_shape] + [torch.ones(x) for x in T.shape[1:]]) 
        self.T_norm = self.T - self.mean_tensor

        ## Tucker Ranks 
        if ranks_tucker is None: # default to full rank
            self.ranks_tucker = self.T_norm.shape 
        else: # Truncate
            self.ranks_tucker = ranks_tucker 

        ## tntorch object. (Handles HOSVD)
        self.t = tn.Tensor(self.T_norm, ranks_tucker=self.ranks_tucker)

        # Apply mode1 product of S with U0. In 4D einsum "ijkl,pi->pjkl" S U0
        self.Q = torch.einsum("i...,ji->j...", self.t.tucker_core(), self.t.Us[0])

        # Mean rank1 parameter tensor
        self.mean_q = [U.mean(0) for i, U in enumerate(self.t.Us) if i>0 ]

    def forward(self, q, rank1 = False):
        if rank1: 
            q = self.q_outer(q) # outer product 
        elif type(q)==list: q=q[0]
        idx = self.get_chars(self.order - 1)
        z = torch.einsum(f"i{idx},{idx}->i", self.Q, q) #"ijkl,jkl->i"
        return z + self.mean_shape
        
    def embed(self, z_true, lr = 0.0001, num_steps = 400, rank1 = True, regularize = False, plot_loss = False): 
        if regularize:
            loss_hist = {"rloss": [] ,"l2loss": [],"sumloss": []}
        else: 
            loss_hist = {"rloss": [] } # ,"l2loss": []}

        def loss_fn():
            r = self.forward(q_hat, rank1 = rank1) - z_true 
            rloss = r@r 
            loss_hist["rloss"].append(rloss.detach().numpy())
            if regularize:
                
                l2loss = sum([torch.norm(q_) for q_ in q_hat])
                loss_hist["l2loss"].append(l2loss.detach())
                
                sumloss = sum([(torch.sum(U@q_) - 1)**2 for q_,U in zip(q_hat,self.t.Us[1:])])
                loss_hist["sumloss"].append(sumloss.detach())

                return rloss + l2loss + sumloss
            else:
                return rloss 

        # initial condition 
        q_hat = [q_.clone() for q_ in self.mean_q]

        # If full rank take outer product q_ijk = q1_i x q2_j x q3_k
        if not rank1:
            idx = self.get_chars(self.order - 1)
            q_hat = [torch.einsum(",".join(idx),*q_hat)]

        for q_ in q_hat: q_.requires_grad = True
    
        # Optimization loop
        optimizer = torch.optim.Adam(q_hat, lr=lr)
        for i in tqdm(range(num_steps)):
            optimizer.zero_grad()    
            loss = loss_fn()
            loss.backward(retain_graph=True)
            optimizer.step()

        q_hat = [q_.detach() for q_ in q_hat]
        
        # plot loss history 
        if plot_loss:
            for k in loss_hist:
                plt.plot(loss_hist[k], label = k)
            plt.legend()

        return q_hat, loss_hist

    def get_q_by_idx(self, idx):
        return [U[j,:] for U, j in zip(self.t.Us[1:],idx)]

    def q_outer(self, q):
        idx = self.get_chars(self.order - 1)
        return torch.einsum(",".join(idx),*q)
    
    @staticmethod
    def get_chars(n) -> str:
        """Get first n characters"""
        return "".join(chr(ord('a') + i) for i in range(n))


class TensorModelPrototypicalEmotions(TensorModel):
    def __init__(self, T, ranks_tucker = None): 
        super().__init__(T, ranks_tucker = ranks_tucker)

        self.expr_label_short = ["AN","DI","FE","HA","SA","SU"]
        self.expr_label = ["Anger","Disgust","Fear","Happiness","Sadness","Surprise"]
        self.expr_dirs = [self.get_dir(expr_idx=i) for i in range(6)]
        
        ## If 3d data is loaded
        if len(T.shape) == 5:
            self.rot_dir = self.get_rotdir()

    def get_dirQ(self, expr_idx = 0):
        q_edit = [x.clone() for x in self.mean_q]
        q_edit[1]  = self.t.Us[2][expr_idx, :]
        return self.q_outer(q_edit)

    def get_rotdirQ(self):
        pn = torch.Tensor([1,-1])/torch.sqrt(torch.Tensor([2]))
        rot_dirQ = torch.Tensor([1,-1]) @ self.t.Us[-1]
        q_edit = [q_.clone() for  q_ in self.mean_q]
        q_edit[-1] = q_edit[-1] + rot_dirQ
        return self.q_outer(q_edit)

    def edit_expressionQ(self, z_true,  expr_idx = 0, strength = 1,rank1=False, **kwargs):
        q_hat, loss_hist = self.embed(z_true, rank1=rank1, **kwargs)
        if not rank1: 
            q_hat = q_hat[0] #Unpack from list
        else:
            q_hat = self.q_outer(q_hat)
        q_dir = strength * self.get_dirQ(expr_idx = expr_idx)
        q_edit = q_hat + q_dir
        z_edit = self.forward(q_edit, rank1 = False)
        return z_edit

    def get_dir(self, expr_idx=0):
        return self.forward(self.get_dirQ(expr_idx=expr_idx)) - self.mean_shape

    def get_rotdir(self):
        return self.forward(self.get_rotdirQ()) - self.mean_shape

    def edit_expression(self,z_true, strength = 1, expr_idx=0):
        return z_true + strength*self.get_dir(expr_idx=expr_idx)

    def edit_rotation(self, z_true, strength = 0.5):
        return z_true + strength*self.get_rotdir()

    def apply_expression(self,z_true, strength = 1, expr_idx=0):
        return z_true + strength*self.expr_dirs[expr_idx]

    def apply_rotation(self, z_true, strength = 0.5):
        return z_true + strength*self.rot_dir
    def save_directions(self, path = "directions/directions.pt"):
        torch.save([self.expr_dirs, self.rot_dir],path)
        
class Manipulator:
    def __init__(self, directions_path= "directions/directions.pt"):
        self.expr_dirs, self.rot_dir = torch.load(directions_path)
        self.expr_label = ["Anger","Disgust","Fear","Happiness","Sadness","Surprise"]

    def apply_expression(self,z_true, strength = 1, expr_idx=0):
        return z_true + strength*self.expr_dirs[expr_idx]

    def apply_rotation(self, z_true, strength = 0.5):
        return z_true + strength*self.rot_dir
