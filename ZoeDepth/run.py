import torch

# Zoe_N
model_zoe_n = torch.hub.load(".", "ZoeD_N", source="local", pretrained=True)

##### sample prediction
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_n.to(DEVICE)


# Local file
from PIL import Image
image_path = '/home/vuser/Documents/rTIC/depth_estimation_video/dataset/deer_robot_converted/cam0/data/0000001120.png'
image = Image.open(image_path).convert("RGB")  # load
depth_numpy = zoe.infer_pil(image)  # as numpy

depth_pil = zoe.infer_pil(image, output_type="pil")  # as 16-bit PIL Image

depth = zoe.infer_pil(image, output_type="tensor")  # as torch tensor
# depth = zoe.infer_pil(image)
print(depth.max())
print(depth.min())
print(depth.mean())
print(depth.std())



# # Tensor 
# from zoedepth.utils.misc import pil_to_batched_tensor
# X = pil_to_batched_tensor(image).to(DEVICE)
# depth_tensor = zoe.infer(X)



# # From URL
# from zoedepth.utils.misc import get_image_from_url

# # Example URL
# URL = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS4W8H_Nxk_rs3Vje_zj6mglPOH7bnPhQitBH8WkqjlqQVotdtDEG37BsnGofME3_u6lDk&usqp=CAU"


# image = get_image_from_url(URL)  # fetch
# depth = zoe.infer_pil(image)

# Save raw
from zoedepth.utils.misc import save_raw_16bit
fpath = "./outputs/output.png"
save_raw_16bit(depth, fpath)

# # Colorize output
# from zoedepth.utils.misc import colorize

# colored = colorize(depth)

# # save colored output
# fpath_colored = "/path/to/output_colored.png"
# Image.fromarray(colored).save(fpath_colored)