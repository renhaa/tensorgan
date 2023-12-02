import torch
import pickle
from matplotlib import pyplot as plt
import  torchvision

def to_np_image(all_images):
    all_images = (all_images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()[0]
    return all_images

def show_torch_img(img):
    img = to_np_image(img)
    plt.imshow(img)
    plt.axis("off")

class StyleGAN():
    
    def __init__(self, path = None, 
                latentspace_type = "w",
                truncation_psi=0.8, 
                truncation_cutoff = 8,
                device = None,
                is_third_time_repo = False,
                transformation_matrix = None
                ) -> None:
        self.path = path
        self.latentspace_type = latentspace_type
        self.truncation_cutoff = truncation_cutoff
        assert self.latentspace_type in ["z","w","wp","s"]
        

        self.truncation_psi = truncation_psi
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else  "cpu"
        else:
            self.device = device
        print("Loading StyleGAN to device", self.device)
        ## inclusiton of latentspace transformation matrix

       
        if transformation_matrix is None:
            self.transformation_matrix = None
        else: 
            self.transformation_matrix = transformation_matrix.to(self.device)
            print("[DEBUG] Sg transform device", self.transformation_matrix.device)
        

        # if is_third_time_repo:
        #     self.G = SG3Generator(checkpoint_path=self.path).decoder
        # else: 
        with open(self.path, 'rb') as f:
            self.G = pickle.load(f)['G_ema'].to(self.device)

        self.res = self.G.img_resolution


        # Stylespace only supported for sg3 currently
        # if self.latentspace_type == "s":
        #     assert "sg3" in self.path or "stylegan3" in self.path
        #     self.G.SDIMS = [s.shape[1] for s in W2S(self.G.synthesis, torch.randn((1,self.G.num_ws,512)).to(self.device)).values()]

        ##for hyperstyle save original params
        self.save_origial_params()
        
    def z_to_w(self, z, to_wp = True):
        w = self.G.mapping(z.unsqueeze(0), None, truncation_psi=self.truncation_psi, truncation_cutoff=self.truncation_cutoff)
        if not to_wp:
            w = w[0,0,:].flatten()
        else: 
            w = w.flatten()
        return w

    def synthesize(self, w, to_np = False, resize = None):
        w = w.to(self.device)
        if self.transformation_matrix is not None:
            w = w @ self.transformation_matrix.T

        # if self.latentspace_type == "s":
        #     img = s2img(self.G, w)
         
        if self.latentspace_type == "z":
            w = self.G.mapping(w.unsqueeze(0), None, truncation_psi=self.truncation_psi, truncation_cutoff=self.truncation_cutoff).flatten()
        
        elif self.latentspace_type == "w":
            w = w.repeat(self.G.num_ws,1).flatten()
        
        w = w.reshape((self.G.num_ws,512)).unsqueeze(0)
        img = self.G.synthesis(w, noise_mode='const', force_fp32=True)
       
        if resize:
            img = torchvision.transforms.Resize((256,256))(img)
        if to_np:
            img = to_np_image(img)

        return img


    def sample(self, seed = None):
        if seed:
            torch.manual_seed(seed)
        
        z = torch.randn([1, self.G.z_dim]).to(device=self.device)  # latent codes
        if self.latentspace_type == "z":
            z = z.squeeze()
            if self.transformation_matrix is not None:
                z = z @ self.transformation_matrix 
            return z
        
        w = self.G.mapping(z, None, truncation_psi=self.truncation_psi, truncation_cutoff=self.truncation_cutoff)
        
        if self.latentspace_type == "w":
            w = w[0,0]
            if self.transformation_matrix is not None:
                w = w @ self.transformation_matrix 
            return w
        elif self.latentspace_type == "wp":
            w = w.flatten()
            if self.transformation_matrix is not None:
                w = w @ self.transformation_matrix 
            return w
        else:
            raise Exception()



    def show(self, w, resize = None):
        w = w.to(self.device)
        img = self.synthesize(w, resize = resize)
        img = to_np_image(img)
        plt.imshow(img)
        plt.axis("off")
        plt.tight_layout()

    def get_mean_latent(self, num_samples = 1000):
        return torch.cat([self.sample().unsqueeze(0) for _ in range(num_samples)]).mean(0) 

    def apply_hyperstyle_weights_deltas(self,weights_deltas):
        params = [p[1] for p in self.G.synthesis.named_parameters() 
                if "weight" in p[0] and not "affine" in p[0] ]
        with torch.no_grad():
            for param, delta in zip(params,weights_deltas):
                if not delta is None:
                    param.copy_(param * (1+delta[0]))
        print("[INFO] Finetuned params set")
    def save_origial_params(self):
            params = [p[1] for p in self.G.synthesis.named_parameters() 
                    if "weight" in p[0] and not "affine" in p[0]]
            hyperstyle_idxs = [5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24]
            self.original_params = [p.to("cpu") if i in hyperstyle_idxs else None for i, p in enumerate(params) ]
            print("[Info] SG original params saved!")

    def reset_params(self):
        params = [p[1] for p in self.G.synthesis.named_parameters() 
                if "weight" in p[0] and not "affine" in p[0] ]
        with torch.no_grad():
            for param, original in zip(params,self.original_params):
                if not original is None:
                    param.copy_(original)
        print("[INFO] Original params restored")



