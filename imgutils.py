
import PIL
from PIL import Image
from PIL import Image, ImageDraw ,ImageFont

import matplotlib.pyplot as plt
import torch
import numpy as np
import os 
import torchvision.transforms as T


def add_margin(pil_img, top = 2, right = 2, bottom = 2, 
                    left = 2, color = (255,255,255)):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    
    result.paste(pil_img, (left, top))
    return result


def show_torch_img(img):
    img = to_np_image(img)
    plt.imshow(img)
    plt.axis("off")

def to_np_image(all_images):
    all_images = (all_images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()[0]
    return all_images

def tensor_to_pil(tensor_imgs):
    if type(tensor_imgs) == list:
        tensor_imgs = torch.cat(tensor_imgs)
    tensor_imgs = (tensor_imgs / 2 + 0.5).clamp(0, 1)
    to_pil = T.ToPILImage()
    pil_imgs = [to_pil(img) for img in tensor_imgs]    
    return pil_imgs

def pil_to_tensor(pil_imgs):
    to_torch = T.ToTensor()
    if type(pil_imgs) == PIL.Image.Image:
        tensor_imgs = to_torch(pil_imgs).unsqueeze(0)*2-1
    elif type(pil_imgs) == list:    
        tensor_imgs = torch.cat([to_torch(pil_imgs).unsqueeze(0)*2-1 for img in pil_imgs]).to(device)
    else:
        raise Exception("Input need to be PIL.Image or list of PIL.Image")
    return tensor_imgs

def add_margin(pil_img, top = 2, right = 2, bottom = 2, 
                    left = 2, color = (255,255,255)):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    
    result.paste(pil_img, (left, top))
    return result

def image_grid(imgs, rows = 1, cols = None, 
                    size = None,
                   titles = None, 
                   top=20,
                   font_size = 20, 
                   text_pos = (0, 0), add_margin_size = None):
    if type(imgs) == list and type(imgs[0]) == torch.Tensor:
        imgs = torch.cat(imgs)
    if type(imgs) == torch.Tensor:
        imgs = tensor_to_pil(imgs)
        
    if not size is None:
        imgs = [img.resize((size,size)) for img in imgs]
    if cols is None:
        cols = len(imgs)
    assert len(imgs) >= rows*cols
    if not add_margin_size is None:
        imgs = [add_margin(img, top = add_margin_size,
                                right = add_margin_size,
                                bottom = add_margin_size, 
                                left = add_margin_size) for img in imgs]
        
   
    w, h = imgs[0].size
    delta = 0
    delta_w = 0
    if len(imgs)> 1 and not imgs[1].size[1] == h:
        delta = h - imgs[1].size[1] #top
        h = imgs[1].size[1]

    if len(imgs)> 1 and not imgs[1].size[0] == w:
        delta_w = w - imgs[1].size[0] #top
        w = imgs[1].size[0]
    if not titles is  None:

        font = ImageFont.truetype(
            # "/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf", 
            "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf", 
                                    size = font_size, encoding="unic")
        h = top + h 
    grid = Image.new('RGB', size=(cols*w+delta_w, rows*h+delta))    
    for i, img in enumerate(imgs):
        
        if not titles is  None:
            text_pos = (int(w/2), 0)
            # print(text_pos)
            img = add_margin(img, top = top, bottom = 0,left=0)
            draw = ImageDraw.Draw(img)
            draw.text(text_pos, titles[i], fill = "black", anchor = "mt",
            font = font)
        if not delta == 0 and i > 0:
           grid.paste(img, box=(i%cols*w, i//cols*h+delta))
        elif not delta_w == 0 and i > 0:
           grid.paste(img, box=(i%cols*w+delta_w, i//cols*h+delta))
        else:
            grid.paste(img, box=(i%cols*w, i//cols*h))
        
    return grid    

def slerp(v0, v1, t, DOT_THRESHOLD=0.9995):
        """helper function to spherically interpolate two arrays v1 v2"""
        if not isinstance(v0, np.ndarray):
            inputs_are_torch = True
            input_device = v0.device
            v0 = v0.cpu().numpy()
            v1 = v1.cpu().numpy()
            t = t.cpu().numpy()
        dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
        if np.abs(dot) > DOT_THRESHOLD:
            v2 = (1 - t) * v0 + t * v1
        else:
            theta_0 = np.arccos(dot)
            sin_theta_0 = np.sin(theta_0)
            theta_t = theta_0 * t
            sin_theta_t = np.sin(theta_t)
            s0 = np.sin(theta_0 - theta_t) / sin_theta_0
            s1 = sin_theta_t / sin_theta_0
            v2 = s0 * v0 + s1 * v1

        if inputs_are_torch:
            v2 = torch.from_numpy(v2).to(input_device)
        return v2

def lerp(z1,z2,t):
    return z1*(1-t) + z2*t



def jupyter_display_video(imgs, tmp_folder = "vid_tmp/", framerate = 4,  out_name = "out.mp4"):

    import os 
    from IPython.display import HTML
    from base64 import b64encode
    from glob import glob
    import shutil
    os.makedirs(tmp_folder,exist_ok=True)
    for i, img in enumerate(imgs):
        img.save(f'{tmp_folder}{i:04}.jpeg')
    cmd_mk_vid = f"ffmpeg -v 1 -y -f image2 -framerate {framerate} -i vid_tmp/%04d.jpeg -c:v libx264 -preset slow -qp 18 -pix_fmt yuv420p {out_name}"
    os.system(cmd_mk_vid)

    shutil.rmtree(tmp_folder)
    mp4 = open(out_name,'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    return HTML("""
    <video width=512 controls>
        <source src="%s" type="video/mp4">
    </video>
    """ % data_url)

def make_vid_from_pil_imgs(imgs, 
                        tmp_folder = "vid_tmp/",
                        framerate = 4,
                        out_name = "out.mp4"):   
    import shutil
    os.makedirs(tmp_folder,exist_ok=True)
    for i, img in enumerate(imgs):
        img.save(f'{tmp_folder}{i:04}.jpeg')
    cmd_mk_vid = f"ffmpeg -v 1 -y -f image2 -framerate {framerate} -i vid_tmp/%04d.jpeg -c:v libx264 -preset slow -qp 18 -pix_fmt yuv420p {out_name}"
    os.system(cmd_mk_vid)
    shutil.rmtree(tmp_folder)