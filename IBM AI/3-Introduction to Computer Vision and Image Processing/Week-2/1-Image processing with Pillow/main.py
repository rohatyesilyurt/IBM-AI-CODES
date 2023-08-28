from PIL import Image
from PIL import ImageOps 

my_image = "lenna.png"

def get_concat_h(im1, im2):
    #https://note.nkmk.me/en/python-pillow-concat-images/
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


image = Image.open(my_image)
# image.show()
# print(image.size) 512x512
print(image.mode) # RGB

im = image.load() # The `Image.open` method does not load image data into the computer memory. The `load` method of `PIL` object reads the file content, decodes it, and expands the image into memory.

image_gray = ImageOps.grayscale(image) # siyah-beyaz haline getirdik
# image_gray.show() 
# print(image_gray.mode) # The mode is `L` for grayscale.


image_gray.quantize(256 // 2) # cut the number of levels in half
# image_gray.show()