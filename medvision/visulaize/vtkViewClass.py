#  Copyright (c) 2020. The Medical Image Computing (MIC) Lab, 陶豪毅
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from vtk.util.vtkImageImportFromArray import *
import vtk
import numpy as np
import os
import pickle


class KeyPressInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, parent, viewer, *args, **kwargs):
        super(KeyPressInteractorStyle).__init__(*args, **kwargs)
        self.parent = vtk.vtkRenderWindowInteractor()
        if parent is not None:
            self.parent = parent

        # print('key press')
        self.AddObserver("KeyPressEvent", viewer.keypressFun)


class MedicalImageWindow:

    def __init__(self):
        self.renWin = vtk.vtkRenderWindow()

        self.renWinInteractor = vtk.vtkRenderWindowInteractor()
        self.renWinInteractor.SetRenderWindow(self.renWin)

        self.renWin.SetSize(450, 300)

        self.camera = vtk.vtkCamera()

        pos = (0, 0, 1, 1)

        self.render = vtk.vtkRenderer()
        self.render.SetBackground(0.8, 0.8, 0.8)
        self.render.SetActiveCamera(self.camera)
        self.render.SetViewport(*pos)

        self.viewer = MedicalImageViewer(self.renWin, self.renWinInteractor, self.render)

    def startWindow(self):

        self.render.ResetCamera()
        self.renWin.AddRenderer(self.render)
        self.renWinInteractor.SetInteractorStyle(KeyPressInteractorStyle(None, self.viewer))
        self.renWin.Render()
        self.renWinInteractor.Start()

        self.renWinInteractor.GetRenderWindow().Finalize()
        self.renWinInteractor.TerminateApp()
        del self.render
        del self.renWin
        del self.renWinInteractor


class MedicalImageViewer:
    def __init__(self, renWin, renWinInteractor, render):
        self.renWin = renWin
        self.renWinInteractor = renWinInteractor
        self.render = render

        self.minGrayValue = 0
        self.maxGrayValue = 10

        self.currentMin, self.currentMax = self.minGrayValue, self.maxGrayValue

        self.volumeProperty_src = vtk.vtkVolumeProperty()
        self.volumeProperty_seg = vtk.vtkVolumeProperty()
        self.src_arr = vtkImageImportFromArray()
        self.seg_arr = vtkImageImportFromArray()
        self.extractVOI_src = vtk.vtkExtractVOI()
        self.extractVOI_seg = vtk.vtkExtractVOI()
        self.dims = []
        self.info = []
        self.slicesActors = []
        self.slicesMappers = []

        self.segOpacity = 0.2

        self.showImage = True
        self.showLabel = True

    def addSrc(self, numpyImage_src, spacing=(1.0, 1.0, 1.0)):
        """
        :param numpyImage_src:
        :param spacing: z,y,x
        :return:
        """
        # print("addSrc")
        self.info.append('addSrc')
        # print('shape of data ', numpyImage_src.shape, "reversed spacing", tuple(reversed(spacing)))

        numpyImage_src = numpyImage_src.astype(np.float32) - np.min(numpyImage_src)
        numpyImage_src = self.maxGrayValue * numpyImage_src / np.max(numpyImage_src)
        # print('minValue, maxValue', self.minGrayValue, self.maxGrayValue)

        self.src_arr.SetArray(numpyImage_src)
        self.src_arr.SetDataSpacing(tuple(reversed(spacing)))
        self.src_arr.SetDataOrigin((0, 0, 0))
        self.src_arr.Update()

        tcfun = vtk.vtkPiecewiseFunction()  # 不透明度传输函数---放在tfun
        tcfun.AddPoint(self.minGrayValue, 0.0)
        tcfun.AddPoint(self.maxGrayValue, 1.0)

        gradtfun = vtk.vtkPiecewiseFunction()  # 梯度不透明度函数---放在gradtfun
        gradtfun.AddPoint(0.0, 0.3)
        gradtfun.AddPoint(0.2, 0.4)
        gradtfun.AddPoint(0.6, 0.6)
        gradtfun.AddPoint(1.0, 1.0)

        ctfun = vtk.vtkColorTransferFunction()  # 颜色传输函数---放在ctfun
        # ctfun.AddRGBPoint(self.minGrayValue, 0.0, 0.0, 0.0)
        # ctfun.AddRGBPoint(self.maxGrayValue, 1.0, 1.0, 1.0)
        ctfun.AddRGBPoint(self.minGrayValue, 0.0, 0.0, 0.0)
        ctfun.AddRGBPoint(self.maxGrayValue, 0.6, 0.6, 0.6)

        outline = vtk.vtkOutlineFilter()
        outline.SetInputConnection(self.src_arr.GetOutputPort())
        outlineMapper = vtk.vtkPolyDataMapper()
        outlineMapper.SetInputConnection(outline.GetOutputPort())
        outlineActor = vtk.vtkActor()
        outlineActor.SetMapper(outlineMapper)

        self.dims = self.src_arr.GetOutput().GetDimensions()
        # print(self.dims)

        self.extractVOI_src.SetInputConnection(self.src_arr.GetOutputPort())
        self.extractVOI_src.SetVOI(0, self.dims[0] - 1, 0, self.dims[1] - 1, 0, self.dims[2] - 1)
        self.extractVOI_src.Update()

        # print(self.extractVOI_src.GetOutput().GetDimensions())

        volumeMapper_src = vtk.vtkGPUVolumeRayCastMapper()
        volumeMapper_src.SetInputData(self.extractVOI_src.GetOutput())

        self.volumeProperty_src.SetColor(ctfun)
        self.volumeProperty_src.SetScalarOpacity(tcfun)
        self.volumeProperty_src.SetGradientOpacity(gradtfun)
        self.volumeProperty_src.SetInterpolationTypeToLinear()
        self.volumeProperty_src.ShadeOn()

        render_volume = vtk.vtkVolume()
        render_volume.SetMapper(volumeMapper_src)
        render_volume.SetProperty(self.volumeProperty_src)

        # self.render.AddActor(outlineActor)
        self.render.AddVolume(render_volume)

    def addGrayScaleSliderToRender(self):
        # print("addGrayScaleSliderToRender")
        self.info.append("addGrayScaleSliderToRender")
        count = 1
        sliderRep_min = vtk.vtkSliderRepresentation2D()
        sliderRep_min.SetMinimumValue(self.minGrayValue)
        sliderRep_min.SetMaximumValue(self.maxGrayValue)
        sliderRep_min.SetValue(self.minGrayValue + 1)
        sliderRep_min.SetTitleText("minValue")
        sliderRep_min.SetSliderLength(0.025)
        sliderRep_min.SetSliderWidth(0.05)
        sliderRep_min.SetEndCapLength(0.005)
        sliderRep_min.SetEndCapWidth(0.025)
        sliderRep_min.SetTubeWidth(0.0125)
        sliderRep_min.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
        sliderRep_min.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
        sliderRep_min.GetPoint1Coordinate().SetValue(1 - 0.05 * count, 0.05)
        sliderRep_min.GetPoint2Coordinate().SetValue(1 - 0.05 * count, 0.45)

        sliderWidget_min = vtk.vtkSliderWidget()
        sliderWidget_min.SetInteractor(self.renWinInteractor)
        sliderWidget_min.SetRepresentation(sliderRep_min)
        sliderWidget_min.SetCurrentRenderer(self.render)
        sliderWidget_min.SetAnimationModeToAnimate()

        sliderRep_max = vtk.vtkSliderRepresentation2D()
        sliderRep_max.SetMinimumValue(self.minGrayValue)
        sliderRep_max.SetMaximumValue(self.maxGrayValue)
        sliderRep_max.SetValue(self.maxGrayValue - 1)
        sliderRep_max.SetTitleText("maxValue")
        sliderRep_max.SetSliderLength(0.025)
        sliderRep_max.SetSliderWidth(0.05)
        sliderRep_max.SetEndCapLength(0.005)
        sliderRep_max.SetEndCapWidth(0.025)
        sliderRep_max.SetTubeWidth(0.0125)
        sliderRep_max.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
        sliderRep_max.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
        sliderRep_max.GetPoint1Coordinate().SetValue(1 - 0.05 * count, 0.55)
        sliderRep_max.GetPoint2Coordinate().SetValue(1 - 0.05 * count, 0.95)

        sliderWidget_max = vtk.vtkSliderWidget()
        sliderWidget_max.SetInteractor(self.renWinInteractor)
        sliderWidget_max.SetRepresentation(sliderRep_max)
        sliderWidget_max.SetCurrentRenderer(self.render)
        sliderWidget_max.SetAnimationModeToAnimate()

        def update_minmax(obj, ev):
            minValue = sliderWidget_min.GetRepresentation().GetValue()
            maxValue = sliderWidget_max.GetRepresentation().GetValue()
            # # reset value
            if minValue >= maxValue:
                if obj == sliderWidget_max:
                    sliderWidget_max.GetRepresentation().SetValue(max(maxValue, minValue + 0.01))
                elif obj == sliderWidget_min:
                    sliderWidget_min.GetRepresentation().SetValue(min(maxValue - 0.01, minValue))
            minValue = sliderWidget_min.GetRepresentation().GetValue()
            maxValue = sliderWidget_max.GetRepresentation().GetValue()

            self.updateGrayScale(minValue, maxValue)

        sliderWidget_min.AddObserver(vtk.vtkCommand.InteractionEvent, update_minmax)
        sliderWidget_max.AddObserver(vtk.vtkCommand.InteractionEvent, update_minmax)

        sliderWidget_min.EnabledOn()
        sliderWidget_max.EnabledOn()

    def addSliceToRender(self):
        # print("addSliceToRender")
        self.info.append("addSliceToRender")
        sliceActor_i_min = vtk.vtkImageSlice()
        sliceMapper_i_min = vtk.vtkImageSliceMapper()
        sliceMapper_i_min.SetInputData(self.extractVOI_src.GetOutput())
        sliceMapper_i_min.SetOrientationToX()
        sliceMapper_i_min.SetSliceNumber(0)
        sliceActor_i_min.SetMapper(sliceMapper_i_min)

        sliceActor_j_min = vtk.vtkImageSlice()
        sliceMapper_j_min = vtk.vtkImageSliceMapper()
        sliceMapper_j_min.SetInputData(self.extractVOI_src.GetOutput())
        sliceMapper_j_min.SetOrientationToY()
        sliceMapper_j_min.SetSliceNumber(0)
        sliceActor_j_min.SetMapper(sliceMapper_j_min)

        sliceActor_k_min = vtk.vtkImageSlice()
        sliceMapper_k_min = vtk.vtkImageSliceMapper()
        sliceMapper_k_min.SetInputData(self.extractVOI_src.GetOutput())
        sliceMapper_k_min.SetOrientationToZ()
        sliceMapper_k_min.SetSliceNumber(0)
        sliceActor_k_min.SetMapper(sliceMapper_k_min)

        sliceActor_i_max = vtk.vtkImageSlice()
        sliceMapper_i_max = vtk.vtkImageSliceMapper()
        sliceMapper_i_max.SetInputData(self.extractVOI_src.GetOutput())
        sliceMapper_i_max.SetOrientationToX()
        sliceMapper_i_max.SetSliceNumber(self.dims[0])
        sliceActor_i_max.SetMapper(sliceMapper_i_max)

        sliceActor_j_max = vtk.vtkImageSlice()
        sliceMapper_j_max = vtk.vtkImageSliceMapper()
        sliceMapper_j_max.SetInputData(self.extractVOI_src.GetOutput())
        sliceMapper_j_max.SetOrientationToY()
        sliceMapper_j_max.SetSliceNumber(self.dims[1])
        sliceActor_j_max.SetMapper(sliceMapper_j_max)

        sliceActor_k_max = vtk.vtkImageSlice()
        sliceMapper_k_max = vtk.vtkImageSliceMapper()
        sliceMapper_k_max.SetInputData(self.extractVOI_src.GetOutput())
        sliceMapper_k_max.SetOrientationToZ()
        sliceMapper_k_max.SetSliceNumber(self.dims[2])
        sliceActor_k_max.SetMapper(sliceMapper_k_max)

        sliceActor_i_min.GetProperty().SetColorLevel(self.maxGrayValue / 2 + self.minGrayValue / 2)
        sliceActor_i_min.GetProperty().SetColorWindow(self.maxGrayValue - self.minGrayValue)
        sliceActor_j_min.GetProperty().SetColorLevel(self.maxGrayValue / 2 + self.minGrayValue / 2)
        sliceActor_j_min.GetProperty().SetColorWindow(self.maxGrayValue - self.minGrayValue)
        sliceActor_k_min.GetProperty().SetColorLevel(self.maxGrayValue / 2 + self.minGrayValue / 2)
        sliceActor_k_min.GetProperty().SetColorWindow(self.maxGrayValue - self.minGrayValue)

        sliceActor_i_max.GetProperty().SetColorLevel(self.maxGrayValue / 2 + self.minGrayValue / 2)
        sliceActor_i_max.GetProperty().SetColorWindow(self.maxGrayValue - self.minGrayValue)
        sliceActor_j_max.GetProperty().SetColorLevel(self.maxGrayValue / 2 + self.minGrayValue / 2)
        sliceActor_j_max.GetProperty().SetColorWindow(self.maxGrayValue - self.minGrayValue)
        sliceActor_k_max.GetProperty().SetColorLevel(self.maxGrayValue / 2 + self.minGrayValue / 2)
        sliceActor_k_max.GetProperty().SetColorWindow(self.maxGrayValue - self.minGrayValue)

        self.render.AddActor(sliceActor_i_min)
        self.render.AddActor(sliceActor_j_min)
        self.render.AddActor(sliceActor_k_min)
        self.render.AddActor(sliceActor_i_max)
        self.render.AddActor(sliceActor_j_max)
        self.render.AddActor(sliceActor_k_max)

        self.slicesActors = [sliceActor_i_min, sliceActor_j_min, sliceActor_k_min,
                             sliceActor_i_max, sliceActor_j_max, sliceActor_k_max]
        self.slicesMappers = [sliceMapper_i_min, sliceMapper_j_min, sliceMapper_k_min,
                              sliceMapper_i_max, sliceMapper_j_max, sliceMapper_k_max]

    def addCropSliderToRender(self):
        # print("addCropSliderToRender")
        self.info.append("addCropSliderToRender")

        def getCropSlider(dim_index, dim_size, render, renWinInteractor):
            sliderRep_min = vtk.vtkSliderRepresentation2D()
            sliderRep_min.SetMinimumValue(0)
            sliderRep_min.SetMaximumValue(dim_size - 1)
            sliderRep_min.SetValue(0)
            sliderRep_min.SetSliderLength(0.025)
            sliderRep_min.SetSliderWidth(0.025)
            sliderRep_min.SetEndCapLength(0.005)
            sliderRep_min.SetEndCapWidth(0.025)
            sliderRep_min.SetTubeWidth(0.0125)
            sliderRep_min.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
            sliderRep_min.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
            sliderRep_min.GetPoint1Coordinate().SetValue(0.05 * dim_index, 0.05)
            sliderRep_min.GetPoint2Coordinate().SetValue(0.05 * dim_index, 0.45)

            sliderWidget_min = vtk.vtkSliderWidget()
            sliderWidget_min.SetInteractor(renWinInteractor)
            sliderWidget_min.SetRepresentation(sliderRep_min)
            sliderWidget_min.SetCurrentRenderer(render)
            sliderWidget_min.SetAnimationModeToAnimate()

            sliderRep_max = vtk.vtkSliderRepresentation2D()
            sliderRep_max.SetMinimumValue(0)
            sliderRep_max.SetMaximumValue(dim_size - 1)
            sliderRep_max.SetValue(dim_size - 1)
            sliderRep_max.SetSliderLength(0.025)
            sliderRep_max.SetSliderWidth(0.025)
            sliderRep_max.SetEndCapLength(0.005)
            sliderRep_max.SetEndCapWidth(0.025)
            sliderRep_max.SetTubeWidth(0.0125)
            sliderRep_max.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
            sliderRep_max.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
            sliderRep_max.GetPoint1Coordinate().SetValue(0.05 * dim_index, 0.55)
            sliderRep_max.GetPoint2Coordinate().SetValue(0.05 * dim_index, 0.95)

            sliderWidget_max = vtk.vtkSliderWidget()
            sliderWidget_max.SetInteractor(renWinInteractor)
            sliderWidget_max.SetRepresentation(sliderRep_max)
            sliderWidget_max.SetCurrentRenderer(render)
            sliderWidget_max.SetAnimationModeToAnimate()

            return sliderWidget_min, sliderWidget_max

        def update_crop(obj, ev):
            # print(obj)
            dim1_minValue = dim1_sliderWidget_min.GetRepresentation().GetValue()
            dim1_maxValue = dim1_sliderWidget_max.GetRepresentation().GetValue()
            dim2_minValue = dim2_sliderWidget_min.GetRepresentation().GetValue()
            dim2_maxValue = dim2_sliderWidget_max.GetRepresentation().GetValue()
            dim3_minValue = dim3_sliderWidget_min.GetRepresentation().GetValue()
            dim3_maxValue = dim3_sliderWidget_max.GetRepresentation().GetValue()
            # # reset value
            if dim1_minValue >= dim1_maxValue:
                if obj == dim1_sliderWidget_max:
                    dim1_sliderWidget_max.GetRepresentation().SetValue(max(dim1_maxValue, dim1_minValue + 1))
                elif obj == dim1_sliderWidget_min:
                    dim1_sliderWidget_min.GetRepresentation().SetValue(min(dim1_maxValue - 1, dim1_minValue))
            if dim2_minValue >= dim2_maxValue:
                if obj == dim2_sliderWidget_max:
                    dim2_sliderWidget_max.GetRepresentation().SetValue(max(dim2_maxValue, dim2_minValue + 1))
                elif obj == dim2_sliderWidget_min:
                    dim2_sliderWidget_min.GetRepresentation().SetValue(min(dim2_maxValue - 1, dim2_minValue))
            if dim3_minValue >= dim3_maxValue:
                if obj == dim3_sliderWidget_max:
                    dim3_sliderWidget_max.GetRepresentation().SetValue(max(dim3_maxValue, dim3_minValue + 1))
                elif obj == dim3_sliderWidget_min:
                    dim3_sliderWidget_min.GetRepresentation().SetValue(min(dim3_maxValue - 1, dim3_minValue))

            dim1_minValue = dim1_sliderWidget_min.GetRepresentation().GetValue()
            dim1_maxValue = dim1_sliderWidget_max.GetRepresentation().GetValue()
            dim2_minValue = dim2_sliderWidget_min.GetRepresentation().GetValue()
            dim2_maxValue = dim2_sliderWidget_max.GetRepresentation().GetValue()
            dim3_minValue = dim3_sliderWidget_min.GetRepresentation().GetValue()
            dim3_maxValue = dim3_sliderWidget_max.GetRepresentation().GetValue()

            # print(dim1_minValue, dim1_maxValue)
            # print(self.dims)
            #
            # print(self.extractVOI_src.GetOutput().GetDimensions())
            # print('update_crop')

            self.updateSlice([dim1_minValue, dim2_minValue, dim3_minValue,
                              dim1_maxValue, dim2_maxValue, dim3_maxValue])

        dim1_sliderWidget_min, dim1_sliderWidget_max = getCropSlider(1, self.dims[0], self.render,
                                                                     self.renWinInteractor)
        dim2_sliderWidget_min, dim2_sliderWidget_max = getCropSlider(2, self.dims[1], self.render,
                                                                     self.renWinInteractor)
        dim3_sliderWidget_min, dim3_sliderWidget_max = getCropSlider(3, self.dims[2], self.render,
                                                                     self.renWinInteractor)

        dim1_sliderWidget_min.AddObserver(vtk.vtkCommand.InteractionEvent, update_crop)
        dim1_sliderWidget_max.AddObserver(vtk.vtkCommand.InteractionEvent, update_crop)
        dim2_sliderWidget_min.AddObserver(vtk.vtkCommand.InteractionEvent, update_crop)
        dim2_sliderWidget_max.AddObserver(vtk.vtkCommand.InteractionEvent, update_crop)
        dim3_sliderWidget_min.AddObserver(vtk.vtkCommand.InteractionEvent, update_crop)
        dim3_sliderWidget_max.AddObserver(vtk.vtkCommand.InteractionEvent, update_crop)

        dim1_sliderWidget_min.EnabledOn()
        dim1_sliderWidget_max.EnabledOn()
        dim2_sliderWidget_min.EnabledOn()
        dim2_sliderWidget_max.EnabledOn()
        dim3_sliderWidget_min.EnabledOn()
        dim3_sliderWidget_max.EnabledOn()

    def addSeg(self, numpyImage_seg, spacing=(1.0, 1.0, 1.0)):
        self.info.append("addSeg")
        # print("add seg")

        # numpyImage_seg = numpyImage_seg.astype(np.int)
        # for i, u in enumerate(np.unique(numpyImage_seg)):
        #     numpyImage_seg[numpyImage_seg == u] = i
        # print(np.unique(numpyImage_seg))

        self.seg_arr.SetArray(numpyImage_seg.astype(np.float))
        self.seg_arr.SetDataSpacing(tuple(reversed(spacing)))
        self.seg_arr.SetDataOrigin((0, 0, 0))
        self.seg_arr.Update()

        tcfun_seg = vtk.vtkPiecewiseFunction()
        tcfun_seg.AddPoint(self.minGrayValue, 0.0)
        tcfun_seg.AddPoint(self.minGrayValue + 0.25, 0.0)
        tcfun_seg.AddPoint(self.maxGrayValue, self.segOpacity)

        gradtfun_seg = vtk.vtkPiecewiseFunction()
        gradtfun_seg.AddPoint(0.05, 0.8)
        gradtfun_seg.AddPoint(0.1, 0.9)
        gradtfun_seg.AddPoint(0.6, 1.0)
        # gradtfun_seg.AddPoint(1.0, 0.9)
        # gradtfun_seg.AddPoint(self.maxGrayValue, 1.1)

        ctfun_seg = vtk.vtkColorTransferFunction()
        ctfun_seg.AddRGBPoint(self.minGrayValue, 0.3, 0.6, 0.3)
        ctfun_seg.AddRGBPoint(self.maxGrayValue, 0.3, 0.6, 0.3)

        # outline = vtk.vtkOutlineFilter()
        # outline.SetInputConnection(self.seg_arr.GetOutputPort())
        # outlineMapper = vtk.vtkPolyDataMapper()
        # outlineMapper.SetInputConnection(outline.GetOutputPort())
        # outlineActor = vtk.vtkActor()
        # outlineActor.SetMapper(outlineMapper)

        self.extractVOI_seg.SetInputConnection(self.seg_arr.GetOutputPort())
        self.extractVOI_seg.SetVOI(0, self.dims[0] - 1, 0, self.dims[1] - 1, 0, self.dims[2] - 1)
        self.extractVOI_seg.Update()

        volumeMapper_seg = vtk.vtkGPUVolumeRayCastMapper()
        volumeMapper_seg.SetInputData(self.extractVOI_seg.GetOutput())

        self.volumeProperty_seg.SetColor(ctfun_seg)
        self.volumeProperty_seg.SetScalarOpacity(tcfun_seg)
        self.volumeProperty_seg.SetGradientOpacity(gradtfun_seg)
        self.volumeProperty_seg.SetInterpolationTypeToLinear()
        self.volumeProperty_seg.ShadeOn()

        render_volume_seg = vtk.vtkVolume()
        render_volume_seg.SetMapper(volumeMapper_seg)
        render_volume_seg.SetProperty(self.volumeProperty_seg)

        sliderRep_segOpacity = vtk.vtkSliderRepresentation2D()
        sliderRep_segOpacity.SetMinimumValue(0)
        sliderRep_segOpacity.SetMaximumValue(1)
        sliderRep_segOpacity.SetValue(self.segOpacity)
        # sliderRep_segOpacity.SetTitleText("seg opacity")
        sliderRep_segOpacity.SetSliderLength(0.025)
        sliderRep_segOpacity.SetSliderWidth(0.05)
        sliderRep_segOpacity.SetEndCapLength(0.005)
        sliderRep_segOpacity.SetEndCapWidth(0.025)
        sliderRep_segOpacity.SetTubeWidth(0.0125)
        sliderRep_segOpacity.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
        sliderRep_segOpacity.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
        sliderRep_segOpacity.GetPoint1Coordinate().SetValue(0.15, 0.05)
        sliderRep_segOpacity.GetPoint2Coordinate().SetValue(0.85, 0.05)

        sliderWidget_segOpacity = vtk.vtkSliderWidget()
        sliderWidget_segOpacity.SetInteractor(self.renWinInteractor)
        sliderWidget_segOpacity.SetRepresentation(sliderRep_segOpacity)
        sliderWidget_segOpacity.SetCurrentRenderer(self.render)
        sliderWidget_segOpacity.SetAnimationModeToAnimate()

        def update_segopacity(obj, ev):
            self.segOpacity = sliderWidget_segOpacity.GetRepresentation().GetValue()

            self.volumeProperty_seg.GetScalarOpacity().AddPoint(self.maxGrayValue, self.segOpacity)

        sliderWidget_segOpacity.AddObserver(vtk.vtkCommand.InteractionEvent, update_segopacity)
        sliderWidget_segOpacity.EnabledOn()

        # self.render.AddActor(outlineActor)
        self.render.AddVolume(render_volume_seg)

    def updateGrayScale(self, minValue, maxValue):
        # print('update_minmax')
        self.currentMin, self.currentMax = minValue, maxValue
        for i in self.info:
            # print(i)
            if i == "addGrayScaleSliderToRender" and self.showImage:
                self.volumeProperty_src.GetScalarOpacity().RemoveAllPoints()
                self.volumeProperty_src.GetScalarOpacity().AddPoint(minValue, 0.0)
                self.volumeProperty_src.GetScalarOpacity().AddPoint(maxValue, 1.0)
            if i == "addSliceToRender":
                for sliceActor in self.slicesActors:
                    sliceActor.GetProperty().SetColorLevel(maxValue / 2 + minValue / 2)
                    sliceActor.GetProperty().SetColorWindow(maxValue - minValue)

    def updateSlice(self, crop):
        if self.slicesMappers:
            assert len(self.slicesMappers) == len(crop)
            for i, s in enumerate(crop):
                self.slicesMappers[i].SetSliceNumber(int(s))

        if "addSrc" in self.info:
            self.extractVOI_src.SetVOI(int(crop[0]), int(crop[3]),
                                       int(crop[1]), int(crop[4]),
                                       int(crop[2]), int(crop[5]))
            self.extractVOI_src.Update()

        if "addSeg" in self.info:
            self.extractVOI_seg.SetVOI(int(crop[0]), int(crop[3]),
                                       int(crop[1]), int(crop[4]),
                                       int(crop[2]), int(crop[5]))
            self.extractVOI_seg.Update()

    def keypressFun(self, obj, event):
        # print(self.info)
        key = self.renWinInteractor.GetKeySym().upper()
        # 键盘控制交互式操作
        # print(key)
        if key == 'L' and "addSeg" in self.info:
            if self.showLabel:
                # print('Hide Label')
                self.volumeProperty_seg.GetScalarOpacity().RemoveAllPoints()
                self.volumeProperty_seg.GetScalarOpacity().AddPoint(self.minGrayValue, 0.0)
                self.volumeProperty_seg.GetScalarOpacity().AddPoint(self.maxGrayValue, 0.0)
            else:
                # print('Show Label')
                self.volumeProperty_seg.GetScalarOpacity().RemoveAllPoints()
                self.volumeProperty_seg.GetScalarOpacity().AddPoint(self.minGrayValue, 0.0)
                self.volumeProperty_seg.GetScalarOpacity().AddPoint(self.minGrayValue + 0.5, 0.0)
                self.volumeProperty_seg.GetScalarOpacity().AddPoint(self.maxGrayValue, self.segOpacity)
            self.showLabel = not self.showLabel

        if key == 'S' and "addSliceToRender" in self.info:
            # print('Slice')
            for sliceActor in self.slicesActors:
                sliceActor.GetProperty().SetOpacity(1 - sliceActor.GetProperty().GetOpacity())

        if key == "T" and "addSrc" in self.info:
            if self.showImage:
                # print('Hide Image')
                self.volumeProperty_src.GetScalarOpacity().RemoveAllPoints()
                self.volumeProperty_src.GetScalarOpacity().AddPoint(self.minGrayValue, 0.0)
                self.volumeProperty_src.GetScalarOpacity().AddPoint(self.maxGrayValue, 0.0)
            else:
                # print('Show Image')
                self.volumeProperty_src.GetScalarOpacity().RemoveAllPoints()
                self.volumeProperty_src.GetScalarOpacity().AddPoint(self.currentMin, 0.0)
                self.volumeProperty_src.GetScalarOpacity().AddPoint(self.currentMax, 1.0)
            self.showImage = not self.showImage

        self.renWin.Render()
        return


def vtkWindowView(numpyImage, numpySeg=None, spacing=(1.0, 1.0, 1.0)):
    m = MedicalImageWindow()
    m.viewer.addSrc(numpyImage, spacing=spacing)
    if numpySeg is not None:
        m.viewer.addSeg(numpySeg, spacing)
    m.viewer.addGrayScaleSliderToRender()
    m.viewer.addSliceToRender()
    m.viewer.addCropSliderToRender()
    m.startWindow()


def vtkWindowViewNotebook(numpyImage, spacing=(1.0, 1.0, 1.0)):
    print('Running vtkWindowViewNotebook ...')
    import sys
    if os.path.exists(os.path.dirname(__file__) + '/tmp.pkl'):
        os.remove(os.path.dirname(__file__) + '/tmp.pkl')
    pickle.dump({'data': numpyImage, 'spacing': spacing}, open(os.path.dirname(__file__) + '/tmp.pkl', 'wb'))
    # print(os.path.dirname(__file__))
    os.system(f'{sys.executable} \"{os.path.dirname(__file__)}/tmp_func.py\" --mode 2')
    print('closing')
