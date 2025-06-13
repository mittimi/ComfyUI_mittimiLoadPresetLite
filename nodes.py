import torch
import toml
import os
import glob
import comfy.sd
import comfy.utils
import comfy.lora
import folder_paths
import hashlib
from server import PromptServer
from aiohttp import web
from PIL import Image, ImageOps, ImageSequence
from PIL.PngImagePlugin import PngInfo
import numpy as np
import json
from comfy.cli_args import args
from nodes import LoraLoader
from .utils import extractLoras, getNewTomlnameExt, load_lora_for_models
import node_helpers
from comfy.comfy_types import IO
import datetime


routes = PromptServer.instance.routes
@routes.post('/mittimiLite_path')
async def my_function(request):
    the_data = await request.post()
    LoadSetParamLiteMittimi.handle_my_message(the_data)
    return web.json_response({})


my_directory_path = os.path.dirname((os.path.abspath(__file__)))
presets_directory_path = os.path.join(my_directory_path, "presets")
preset_list = []
tmp_list = []
tmp_list += glob.glob(f"{presets_directory_path}/**/*.toml", recursive=True)
for l in tmp_list:
    preset_list.append(os.path.relpath(l, presets_directory_path))
if len(preset_list) > 1: preset_list.sort()
vae_new_list = folder_paths.get_filename_list("vae")
vae_new_list.insert(0,"Use_merged_vae")


class LoadSetParamLiteMittimi:

    def __init__(self):
        self.device = comfy.model_management.intermediate_device()
        self.loaded_lora = None
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "checkpoint": (folder_paths.get_filename_list("checkpoints"),),
                "clip_num": ("INT", {"default": -1, "min": -24, "max": -1} ),
                "vae": (vae_new_list,),
                "positive_A": ("STRING", {"multiline": True}),
                "positive_B": ("STRING", {"multiline": True}),
                "positive_C": ("STRING", {"multiline": True}),
                "negative_A": ("STRING", {"multiline": True}),
                "negative_B": ("STRING", {"multiline": True}),
                "negative_C": ("STRING", {"multiline": True}),
                "width": ("INT", {"default": 512, "min": 8, "max": 16384}),
                "height": ("INT", {"default": 512, "min": 8, "max": 16384}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 999999}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 999999}),
                "cfg": ("FLOAT", ),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "seed": ("INT", {"default": 1, "min": 0, "max": 0xffffffffffffffff}),
                "preset": (preset_list,),
            },
            "hidden": {"node_id": "UNIQUE_ID" }
        }

    RETURN_TYPES = (folder_paths.get_filename_list("checkpoints"), "INT", vae_new_list, "STRING", "STRING", "INT", "FLOAT", comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS, "INT", "INT", "INT", "INT", "PDATA", )
    RETURN_NAMES = ("checkpoint_name", "clip_num", "vae_name", "positive_prompt_text", "negative_prompt_text", "steps", "cfg", "sampler_name", "scheduler", "seed", "width", "height", "batch_size", "parameters_data", )
    FUNCTION = "loadAndSettingParameters03"
    CATEGORY = "mittimiTools"
    NAME = "name"

    def loadAndSettingParameters03(self, checkpoint, clip_num, vae, positive_A, positive_B, positive_C, negative_A, negative_B, negative_C, width, height, batch_size, steps, cfg, sampler_name, scheduler, seed, node_id, preset=[], ):

        # ckpt_path = folder_paths.get_full_path("checkpoints", checkpoint)
        # out3 = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        # re_ckpt = out3[0]
        # re_clip = out3[1].clone()
        # re_clip.clip_layer(clip_num)
        # re_vae = out3[2]
        # if (vae != "Use_merged_vae"):
        #     vae_path = folder_paths.get_full_path("vae", vae)
        #     sd = comfy.utils.load_torch_file(vae_path)
        #     re_vae = comfy.sd.VAE(sd=sd)
        
        poscombtxt = positive_A + positive_B + positive_C
        negcombtxt = negative_A + negative_B + negative_C
        
        parameters_data = []
        parameters_data.append( {
            'checkpoint':checkpoint,
            'clip':clip_num, 
            'vae':vae, 
            'posA':positive_A,
            'posB':positive_B,
            'posC':positive_C,
            'negA':negative_A,
            'negB':negative_B,
            'negC':negative_C,
            'width':width, 
            'height':height, 
            'batchsize':batch_size,
            'step':steps, 
            'cfg':cfg, 
            'sampler':sampler_name, 
            'scheduler':scheduler, 
            'seed':seed, 
            } )
        
        return(checkpoint, clip_num, vae, poscombtxt, negcombtxt, steps, cfg, sampler_name, scheduler, seed, width, height, batch_size, parameters_data, )

    def handle_my_message(d):
        
        preset_data = ""
        preset_path = os.path.join(presets_directory_path, d['message'])
        with open(preset_path, 'r') as f:
            preset_data = toml.load(f)
        PromptServer.instance.send_sync("my.custom.message", {"message":preset_data, "node":d['node_id']})




class LoadCheckpointLiteMittimi:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "checkpoint_name": (folder_paths.get_filename_list("checkpoints"),),
                "clip_num": ("INT", {"default": -1, "min": -24, "max": -1} ),
                "vae_name": (vae_new_list,),
            }
        }
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", )
    RETURN_NAMES = ("model", "clip", "vae", )
    FUNCTION = "loadCheckpointLiteMittimi"
    CATEGORY = "mittimiTools"

    def loadCheckpointLiteMittimi(self, checkpoint_name, clip_num, vae_name, ):

        ckpt_path = folder_paths.get_full_path("checkpoints", checkpoint_name)
        out3 = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        re_ckpt = out3[0]
        re_clip = out3[1].clone()
        re_clip.clip_layer(clip_num)
        re_vae = out3[2]
        if (vae_name != "Use_merged_vae"):
            vae_path = folder_paths.get_full_path("vae", vae_name)
            sd = comfy.utils.load_torch_file(vae_path)
            re_vae = comfy.sd.VAE(sd=sd)
        
        return(re_ckpt, re_clip, re_vae, )





class TextToConditioningLiteMittimi:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "clip": (IO.CLIP, ),
                "positive_prompt_text": ("STRING", ),
                "negative_prompt_text": ("STRING", ),
                "seed": ("INT", {"default": 1, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    RETURN_TYPES = ("MODEL", "CLIP", "CONDITIONING", "CONDITIONING", )
    RETURN_NAMES = ("model", "clip", "conditioning_positive", "conditioning_negative", )
    FUNCTION = "textToConditioningLiteMittimi"
    CATEGORY = "mittimiTools"

    def textToConditioningLiteMittimi(self, model, clip, positive_prompt_text, negative_prompt_text, seed, ):

        plora = []
        ptxt, plora = extractLoras(positive_prompt_text)

        if len(plora):
            for lora in plora:
                if lora['vector']:
                    model, clip, populated_vector = load_lora_for_models(model, clip, lora['fullpath'], lora['strength'], lora['strength'], seed, lora['vector'])
                else:
                    model, clip = LoraLoader().load_lora(model, clip, lora['path'], lora['strength'], lora['strength'])
                    
        ptokens = clip.tokenize(ptxt)
        ntokens = clip.tokenize(negative_prompt_text)
        poutput = clip.encode_from_tokens(ptokens, return_pooled=True, return_dict=True)
        noutput = clip.encode_from_tokens(ntokens, return_pooled=True, return_dict=True)
        pcond = poutput.pop("cond")
        ncond = noutput.pop("cond")

        return(model, clip, [[pcond, poutput]], [[ncond, noutput]], )














class CombineParamDataLiteMittimi:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "param1": ("PDATA", ),
            },
            "optional": {
                "param2": ("PDATA", ),
            },
        }
    RETURN_TYPES = ("PDATA", )
    RETURN_NAMES = ("parameters_data", )
    FUNCTION = "combineParamDataLiteMittimi"
    CATEGORY = "mittimiTools"

    def combineParamDataLiteMittimi(self, param1, param2=None, ):

        return_param = param1[:]
        if param2:
            return_param += param2[:]
        
        return(return_param, )


class SaveImageParamLiteMittimi:

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4
        
    @classmethod
    def INPUT_TYPES(s):

        file_type = ['PNG', 'WEBP']        
        return {
            "required": {
                "images": ("IMAGE", ),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "type": (file_type, ),
                "level_or_method": ("INT", {"default":6, "min":1, "max":9}),
                "lossless": ("BOOLEAN", {"default":True}),
            },
            "optional": {
                "quality": ("INT", {"default": 80, "min": 1, "max": 100}),
                "parameters_data": ("PDATA", ), 
                "add_text": ("STRING", ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }
    
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "saveimageMittimi"
    CATEGORY = "mittimiTools"

    def saveimageMittimi(self, images, filename_prefix, type, level_or_method, lossless, quality=80, parameters_data=None, add_text="", prompt=None, extra_pnginfo=None, ):
        
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])

        results = list()
        bseed = 0
        if parameters_data:
            bseed = parameters_data[0]['seed'] if "seed" in parameters_data[0] else 0
        if bseed > 0:
            bseed = "_"+str(bseed)
        else:
            dt_now = datetime.datetime.now()
            bseed = dt_now.strftime('_%Y%m%d%H%M%S')

        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            parameters_text = ""

            if add_text != "":
                parameters_text += f"{add_text}\n\n"

            if parameters_data is not None:
                param_counter = 0
                for pd in parameters_data:
                    if param_counter > 0: parameters_text += "\n\n"
                    parameters_text += f"{pd['posA']+pd['posB']+pd['posC']}\nNegative prompt: {pd['negA']+pd['negB']+pd['negC']}\nSteps: {pd['step']}, Sampler: {pd['sampler']}, Scheduler: {pd['scheduler']}, CFG Scale: {pd['cfg']}, Seed: {pd['seed']}, Batch count: {batch_number}, Size: {pd['width']}x{pd['height']}, Model: {pd['checkpoint']}, Clip: {pd['clip']}, VAE: {pd['vae']}"
                    param_counter += 1
                parameters_text += ", Version: ComfyUI"
                
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            rseed = bseed if batch_number == 0 else f"{bseed}_{batch_number:03}"
            
            if type =="WEBP":
                
                metadata = img.getexif()
                metadata[270] = f"parameters:{parameters_text}"
                
                if prompt is not None:
                    metadata[272] = "prompt:{}".format(json.dumps(prompt))

                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata[271] = "{}:{}".format(x, json.dumps(extra_pnginfo[x]))
                        
                file = f"{filename_with_batch_num}_{counter:05}{rseed}.webp"
                method = level_or_method if level_or_method <7 else 6
                img.save(os.path.join(full_output_folder, file), "webp", exif=metadata, lossless=lossless, quality=quality, method=method)
            
            else:
                metadata = None
                if not args.disable_metadata:
                    metadata = PngInfo()
                metadata.add_text("parameters", parameters_text)
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                
                file = f"{filename_with_batch_num}_{counter:05}{rseed}.png"
                img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=level_or_method)

            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "images": results } }
    

class SaveParamToPresetLiteMittimi:
    @classmethod
    def INPUT_TYPES(s):

        savetype_list = ["new save", "overwrite save"]

        return {
            "required": {
                "param": ("PDATA", ),
                "tomlname": ("STRING", {"default": "new_preset"}),
                "savetype": (savetype_list,),
            },
        }
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "saveparamtopresetLiteMittimi"
    CATEGORY = "mittimiTools"
    
    def saveparamtopresetLiteMittimi(self, param, tomlname, savetype, ):

        tomltext = f"CheckpointName = \"{param[0]['checkpoint']}\"\n"
        tomltext += f"ClipSet = {param[0]['clip']}\n"
        tomltext += f"VAE = \"{param[0]['vae']}\"\n"
        tomltext += f"PositivePromptA = \"{param[0]['posA']}\"\n"
        tomltext += f"PositivePromptB = \"{param[0]['posB']}\"\n"
        tomltext += f"PositivePromptC = \"{param[0]['posC']}\"\n"
        tomltext += f"NegativePromptA = \"{param[0]['negA']}\"\n"
        tomltext += f"NegativePromptB = \"{param[0]['negB']}\"\n"
        tomltext += f"NegativePromptC = \"{param[0]['negC']}\"\n"
        tomltext += f"Width = {param[0]['width']}\n"
        tomltext += f"Height = {param[0]['height']}\n"
        tomltext += f"BatchSize = {param[0]['batchsize']}\n"
        tomltext += f"Steps = {param[0]['step']}\n"
        tomltext += f"CFG = {param[0]['cfg']}\n"
        tomltext += f"SamplerName = \"{param[0]['sampler']}\"\n"
        tomltext += f"Scheduler = \"{param[0]['scheduler']}\"\n"
        
        tomlnameExt = getNewTomlnameExt(tomlname, presets_directory_path, savetype)
        
        check_folder_path = os.path.dirname(f"{presets_directory_path}/{tomlnameExt}")
        os.makedirs(check_folder_path, exist_ok=True)

        with open(f"{presets_directory_path}/{tomlnameExt}", mode='w') as f:
            f.write(tomltext)

        return()


class LoadImageParamLiteMittimi:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {"required":
                {"image": (sorted(files), {"image_upload": True}), },                
                }
    
    CATEGORY = "mittimiTools"

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "STRING", )
    RETURN_NAMES = ("image", "mask", "width", "height", "parameters_text", )
    FUNCTION = "loadimageParamLiteMittimi"

    def loadimageParamLiteMittimi(self, image, ):

        image_path = folder_paths.get_annotated_filepath(image)
        img = node_helpers.pillow(Image.open, image_path)

        parameters = ""

        if img.format == "PNG":
            print("It's PNG.")
            png_info = img.info
            if png_info is not None:
                print("info founded.")
                parameters = png_info.get('parameters',"")
        
        elif img.format == 'WEBP':
            print("It's WEBP.")
            webp_info = img.getexif()
            parameters = webp_info.get(270,"").replace("parameters:", "", 1)
        
        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]
            
            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            elif i.mode == 'P' and 'transparency' in i.info:
                mask = np.array(i.convert('RGBA').getchannel('A')).astype(np.float32) /255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))
        
        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]
        
        return (output_image, output_mask, w, h, parameters, )
    
    @classmethod
    def IS_CHANGED(s, image, ):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image, ):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True


NODE_CLASS_MAPPINGS = {
    "LoadSetParamLiteMittimi": LoadSetParamLiteMittimi,
    "LoadCheckpointLiteMittimi": LoadCheckpointLiteMittimi,
    "TextToConditioningLiteMittimi": TextToConditioningLiteMittimi,
    "SaveImageParamLiteMittimi": SaveImageParamLiteMittimi,
    "CombineParamDataLiteMittimi": CombineParamDataLiteMittimi,
    "SaveParamToPresetLiteMittimi": SaveParamToPresetLiteMittimi,
    "LoadImageParamLiteMittimi": LoadImageParamLiteMittimi,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadSetParamLiteMittimi": "LoadSetParametersLite",
    "LoadCheckpointLiteMittimi": "LoadCheckpointLite",
    "TextToConditioningLiteMittimi": "TextToConditioningLite",
    "SaveImageParamLiteMittimi": "SaveImageWithParamTextLite",
    "CombineParamDataLiteMittimi": "CombineParamDataLite",
    "SaveParamToPresetLiteMittimi": "SaveParamToPresetLite",
    "LoadImageParamLiteMittimi": "LoadImageParamLite"
}