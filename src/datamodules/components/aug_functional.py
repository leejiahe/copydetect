import os
import random
import string
import numpy as np

from PIL import Image

import augly
import augly.image as imaugs

import pilgram

from src.utils import get_image_paths

randomRGB = lambda: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

letters = string.ascii_letters + string.digits + string.punctuation
letters = [letter for letter in letters]
randomText = lambda: (''.join(random.sample(letters, k = random.randint(3, 6))))

FONT_PATHS = [os.path.join(augly.utils.FONTS_DIR, f) for f in os.listdir(augly.utils.FONTS_DIR) \
              if f.endswith('ttf') and not f.startswith('Noto')] # There is error using the noto font library

class MemeRandomFormat:
    def __call__(self, image):
        text = randomText()
        font_file = random.choice(FONT_PATHS)
        opacity = random.uniform(0.5, 1)
        text_color = randomRGB()
        caption_height = random.randint(50, 200)
        meme_bg_color = randomRGB()
        
        try:
            return imaugs.aug_np_wrapper(image,
                                         imaugs.meme_format,
                                         **{'text': text,
                                            'font_file': font_file,
                                            'opacity': opacity,
                                            'text_color': text_color,
                                            'caption_height': caption_height,
                                            'meme_bg_color': meme_bg_color,
                                           }
                                        )
        except:
            raise Exception("Error for MemeRandomFormat") 
            return image
        
        
        
class OverlayRandomStripes:
    def __call__(self,image):
        
        line_width = random.uniform(0.1, 0.8)
        line_color = randomRGB()
        line_angle = random.randrange(-90, 90)
        line_density = random.uniform(0.5, 1)
        line_type = random.choice(augly.utils.SUPPORTED_LINE_TYPES)
        line_opacity = random.uniform(0.5, 1)
        try:
            return imaugs.aug_np_wrapper(image,
                                         imaugs.overlay_stripes,
                                         **{'line_width':line_width,
                                            'line_color':line_color,
                                            'line_angle':line_angle,
                                            'line_density':line_density,
                                            'line_type':line_type,
                                            'line_opacity':line_opacity,
                                           }
                                        )
        except:
            raise Exception("Error for OverlayRandomStripes") 
            return image



class OverlayRandomEmoji:
    def __init__(self):
        self.emoji_paths = []
        for folder in os.listdir(augly.utils.EMOJI_DIR):
            files_path = [os.path.join(augly.utils.EMOJI_DIR, folder, file) for file in os.listdir(os.path.join(augly.utils.EMOJI_DIR, folder))]
            self.emoji_paths.extend(files_path)

    def __call__(self, image):
        emoji_path = random.choice(self.emoji_paths)
        opacity = random.uniform(0.4, 1)
        emoji_size = random.uniform(0.4, 0.8)
        x_pos = random.uniform(0, 0.75)
        y_pos = random.uniform(0, 0.75)
                        
        try:
            return imaugs.aug_np_wrapper(image,
                                         imaugs.overlay_emoji,
                                         **{'emoji_path':emoji_path,
                                            'opacity':opacity,
                                            'emoji_size':emoji_size,
                                            'x_pos':x_pos,
                                            'y_pos':y_pos,
                                           }
                                        )
        except:
            raise Exception("Error for OverlayRandomEmoji") 
            return image
        
        
        
class OverlayRandomText:
    def __call__(self, image):
        text_indices = [[random.randint(0, 1000) for _ in range(random.randint(5, 10))] \
                        for _ in range(random.randint(1, 3))]
        font_path = random.choice(FONT_PATHS)
        font_size = random.uniform(0.1, 0.4)
        opacity = random.uniform(0.5, 1)
        color = randomRGB()
        x_pos = random.uniform(0, 0.6)
        y_pos = random.uniform(0, 0.6)
                                
        try:
            return imaugs.aug_np_wrapper(image,
                                         imaugs.overlay_text,
                                         **{'text':text_indices,
                                            'font_file':font_path,
                                            'font_size':font_size,
                                            'opacity':opacity,
                                            'color':color,
                                            'x_pos':x_pos,
                                            'y_pos':y_pos,
                                         }
                                        )
        except:
            raise Exception("Error for OverlayRandomText") 
            return image
        
        
        
class OverlayOntoRandomBackgroundImage:
    def __init__(self, bg_image_dir:str):
        self.path, self.bg_image = get_image_paths(bg_image_dir)
        self.bg_image = random.sample(self.bg_image, 50000)
    
    def __call__(self, image):
        background_image = random.choice(self.bg_image)
        background_image = Image.open(os.path.join(self.path, background_image))
        opacity = random.uniform(0.8, 1)
        overlay_size = random.uniform(0.3, 0.5)
        x_pos = random.uniform(0, 0.4)
        y_pos = random.uniform(0, 0.4)
                            
        try:
            return imaugs.aug_np_wrapper(image,
                                         imaugs.overlay_onto_background_image,
                                         **{'background_image':background_image,
                                            'opacity':opacity,
                                            'overlay_size':overlay_size,
                                            'x_pos':x_pos,
                                            'y_pos':y_pos,
                                           }
                                        )
        except:
            raise Exception("Error for OverlayOntoRandomBackgroundImage") 
            return image
        
        
class OverlayOntoRandomForegroundImage:
    def __init__(self, fg_image_dir:str):
        self.path, self.fg_image = get_image_paths(fg_image_dir)
        self.fg_image = random.sample(self.fg_image, 50000)
        
    def __call__(self, image):
        foreground_image = random.choice(self.fg_image)
        foreground_image = Image.open(os.path.join(self.path, foreground_image))
        foreground_image = np.array(foreground_image)
        image = Image.fromarray(image)
        opacity = random.uniform(0.8, 1)
        overlay_size = random.uniform(0.3, 0.5)
        x_pos = random.uniform(0, 0.4)
        y_pos = random.uniform(0, 0.4)
                            
        try:
            return imaugs.aug_np_wrapper(foreground_image,
                                         imaugs.overlay_onto_background_image,
                                         **{'background_image':image,
                                            'opacity':opacity,
                                            'overlay_size':overlay_size,
                                            'x_pos':x_pos,
                                            'y_pos':y_pos,
                                           }
                                        )
        except:
            raise Exception("Error for OverlayOntoRandomForegroundImage") 
            return image
        
        
        
class OverlayOntoRandomScreenshot:
    def __init__(self):
        self.template_bboxes_filepath = augly.utils.BBOXES_PATH
        self.template_filepath = [os.path.join(augly.utils.SCREENSHOT_TEMPLATES_DIR, f) \
                                  for f in os.listdir(augly.utils.SCREENSHOT_TEMPLATES_DIR) \
                                  if f.endswith(('png', 'jpg'))]

    def __call__(self, image):
        template_filepath = random.choice(self.template_filepath)
        template_bboxes_filepath = self.template_bboxes_filepath
        crop_src_to_fit = True
        resize_src_to_match_template = True
                            
        try:
            return imaugs.aug_np_wrapper(image,
                                         imaugs.overlay_onto_screenshot,
                                         **{'template_filepath':template_filepath,
                                            'template_bboxes_filepath':template_bboxes_filepath,
                                            'crop_src_to_fit':crop_src_to_fit,
                                            'resize_src_to_match_template':resize_src_to_match_template,
                                         }
                                        )
        except:
            raise Exception("Error for OverlayOntoRandomScreenshot")
            return image
        
        
        
class ApplyIGFilter:
    def __init__(self):
        self.ig_filters = ['_1977', 'brooklyn', 'clarendon', 'earlybird', 'inkwell',
                           'kelvin', 'maven', 'moon', 'nashville', 'reyes', 'rise',
                           'slumber', 'toaster', 'valencia', 'willow', 'xpro2']
        
    def __call__(self, image):
        image = Image.fromarray(image)
        ig_filter = np.random.choice(self.ig_filters)
        try:
            func = eval(f'pilgram.{ig_filter}')
            return np.array(func(image))
        except:
            raise Exception(f"Error for ApplyIGFilter {ig_filter}")
            return image
        
        
        
class BlendImage:
    def __init__(self, bg_image_dir: str):
        self.css_blends = ['color', 'color_burn', 'darken', 'difference', 'exclusion',
                           'hue', 'lighten', 'multiply', 'screen']
        self.path, self.bg_image = get_image_paths(bg_image_dir)
        self.bg_image = random.sample(self.bg_image, 50000)
        
    def __call__(self, image):
        image = Image.fromarray(image)
        css_blends = np.random.choice(self.css_blends)
        background_image = random.choice(self.bg_image)
        background_image = Image.open(os.path.join(self.path, background_image))

        try:
            func = eval(f'pilgram.css.blending.{css_blends}')
            return np.array(func(image, background_image))
        except:
            raise Exception(f"Error for BlendImage {css_blends}")
            return image