from medvision.io import ImageIO
from medvision.visulaize import vtkWindowView, vtkWindowViewNotebook, vtkShow, vtkShowNotebook, \
    vtkShowTogether, vtkShowMulti, vtkScreenshot

if __name__ == '__main__':
    data, _, spacing, origin = ImageIO.loadArray('../samples/lung.nii.gz')
    mask, _, _, _ = ImageIO.loadArray('../samples/lung_mask.nii.gz')
    seg, _, _, _ = ImageIO.loadArray('../samples/nodule_seg.nii.gz')

    # vtkWindowViewNotebook(data[0], spacing) or

    # vtkShowNotebook(data[0], spacing) or vtkShow

    vtkShowTogether(data[0], mask[0], spacing=spacing)
    vtkShowMulti(data[0], [mask[0], seg[0]], spacing=spacing)

    # you should run the following in __main__
    # vtkScreenshot('vtkScreenshot.png', data[0], spacing)
