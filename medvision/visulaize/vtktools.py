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
import threading
import multiprocessing
import pickle


def getRenderOfSrcImage(count, camera, renWinInteractor, numpy_image_src, spacing,
                        minValue=0, maxValue=10, pos=(0, 0, 1.0, 1.0)):
    numpy_image_src = numpy_image_src.astype(np.float32) - np.min(numpy_image_src)
    numpy_image_src = maxValue * numpy_image_src / np.max(numpy_image_src)
    print('minValue, maxValue', minValue, maxValue)

    render = vtk.vtkRenderer()
    render.SetBackground(0.8, 0.8, 0.8)
    render.SetActiveCamera(camera)
    render.SetViewport(*pos)

    img_arr = vtkImageImportFromArray()
    img_arr.SetArray(np.ascontiguousarray(numpy_image_src))
    img_arr.SetDataSpacing(spacing)
    img_arr.SetDataOrigin((0, 0, 0))
    img_arr.Update()

    tcfun = vtk.vtkPiecewiseFunction()  # 不透明度传输函数---放在tfun
    tcfun.AddPoint(minValue, 0.0)
    tcfun.AddPoint(maxValue, 1.0)

    gradtfun = vtk.vtkPiecewiseFunction()  # 梯度不透明度函数---放在gradtfun
    gradtfun.AddPoint(0, 0)
    gradtfun.AddPoint(0.2, 0.1)
    gradtfun.AddPoint(0.6, 0.5)
    gradtfun.AddPoint(1.0, 1.0)

    ctfun = vtk.vtkColorTransferFunction()  # 颜色传输函数---放在ctfun
    ctfun.AddRGBPoint(minValue, 0.0, 0.0, 0.0)
    ctfun.AddRGBPoint(maxValue, 0.6, 0.6, 0.6)

    # outline = vtk.vtkOutlineFilter()
    # outline.SetInputConnection(img_arr.GetOutputPort())
    # outlineMapper = vtk.vtkPolyDataMapper()
    # outlineMapper.SetInputConnection(outline.GetOutputPort())
    # outlineActor = vtk.vtkActor()
    # outlineActor.SetMapper(outlineMapper)

    volume_mapper_src = vtk.vtkGPUVolumeRayCastMapper()
    volume_mapper_src.SetInputConnection(img_arr.GetOutputPort())

    volume_property = vtk.vtkVolumeProperty()
    volume_property.SetColor(ctfun)
    volume_property.SetScalarOpacity(tcfun)
    volume_property.SetGradientOpacity(gradtfun)
    volume_property.SetInterpolationTypeToLinear()
    volume_property.ShadeOn()

    render_volume = vtk.vtkVolume()
    render_volume.SetMapper(volume_mapper_src)
    render_volume.SetProperty(volume_property)

    # render.AddActor(outlineActor)
    render.AddVolume(render_volume)
    render.ResetCamera()

    sliderRep_min = vtk.vtkSliderRepresentation2D()
    sliderRep_min.SetMinimumValue(0)
    sliderRep_min.SetMaximumValue(10)
    sliderRep_min.SetValue(1)
    sliderRep_min.SetTitleText("minValue")
    sliderRep_min.SetSliderLength(0.025)
    sliderRep_min.SetSliderWidth(0.05)
    sliderRep_min.SetEndCapLength(0.005)
    sliderRep_min.SetEndCapWidth(0.025)
    sliderRep_min.SetTubeWidth(0.0125)
    sliderRep_min.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
    sliderRep_min.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
    sliderRep_min.GetPoint1Coordinate().SetValue(0.15 / count, 0.1)
    sliderRep_min.GetPoint2Coordinate().SetValue(0.45 / count, 0.1)

    sliderWidget_min = vtk.vtkSliderWidget()
    sliderWidget_min.SetInteractor(renWinInteractor)
    sliderWidget_min.SetRepresentation(sliderRep_min)
    sliderWidget_min.SetCurrentRenderer(render)
    sliderWidget_min.SetAnimationModeToAnimate()

    sliderRep_max = vtk.vtkSliderRepresentation2D()
    sliderRep_max.SetMinimumValue(0)
    sliderRep_max.SetMaximumValue(10)
    sliderRep_max.SetValue(9)
    sliderRep_max.SetTitleText("maxValue")
    sliderRep_max.SetSliderLength(0.025)
    sliderRep_max.SetSliderWidth(0.05)
    sliderRep_max.SetEndCapLength(0.005)
    sliderRep_max.SetEndCapWidth(0.025)
    sliderRep_max.SetTubeWidth(0.0125)
    sliderRep_max.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
    sliderRep_max.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
    sliderRep_max.GetPoint1Coordinate().SetValue(0.55 / count, 0.1)
    sliderRep_max.GetPoint2Coordinate().SetValue(0.85 / count, 0.1)

    sliderWidget_max = vtk.vtkSliderWidget()
    sliderWidget_max.SetInteractor(renWinInteractor)
    sliderWidget_max.SetRepresentation(sliderRep_max)
    sliderWidget_max.SetCurrentRenderer(render)
    sliderWidget_max.SetAnimationModeToAnimate()

    def update_minmax(obj, ev):
        # print(obj)
        minValue = sliderWidget_min.GetRepresentation().GetValue()
        maxValue = sliderWidget_max.GetRepresentation().GetValue()
        # reset value
        if minValue >= maxValue:
            if obj == sliderWidget_max:
                sliderWidget_max.GetRepresentation().SetValue(max(maxValue, minValue + 0.01))
            elif obj == sliderWidget_min:
                sliderWidget_min.GetRepresentation().SetValue(min(maxValue - 0.01, minValue))
        minValue = sliderWidget_min.GetRepresentation().GetValue()
        maxValue = sliderWidget_max.GetRepresentation().GetValue()

        tcfun.RemoveAllPoints()
        tcfun.AddPoint(minValue, 0.0)
        tcfun.AddPoint(maxValue, 1.0)
        volume_property.SetScalarOpacity(tcfun)

    sliderWidget_min.AddObserver(vtk.vtkCommand.InteractionEvent, update_minmax)
    sliderWidget_max.AddObserver(vtk.vtkCommand.InteractionEvent, update_minmax)
    sliderWidget_min.EnabledOn()
    sliderWidget_max.EnabledOn()
    return render, sliderWidget_min, sliderWidget_max


def getRenderOfSegImage(count, camera, renWinInteractor, numpy_image_seg, spacing,
                        minValue=0, maxValue=10, pos=(0, 0, 1.0, 1.0)):
    numpy_image_seg = numpy_image_seg.astype(np.float32) - np.min(numpy_image_seg)
    numpy_image_seg = maxValue * numpy_image_seg / np.max(numpy_image_seg)
    print('minValue, maxValue', minValue, maxValue)

    render = vtk.vtkRenderer()
    render.SetBackground(0.8, 0.8, 0.8)
    render.SetActiveCamera(camera)
    render.SetViewport(*pos)

    img_arr_seg = vtkImageImportFromArray()
    img_arr_seg.SetArray(np.ascontiguousarray(numpy_image_seg))
    img_arr_seg.SetDataSpacing(spacing)
    img_arr_seg.SetDataOrigin((0, 0, 0))
    img_arr_seg.Update()

    tcfun_seg = vtk.vtkPiecewiseFunction()
    tcfun_seg.AddPoint(minValue, 0.0)
    tcfun_seg.AddPoint(maxValue, 1.0)

    gradtfun_seg = vtk.vtkPiecewiseFunction()
    gradtfun_seg.AddPoint(0, 0)
    gradtfun_seg.AddPoint(0.2, 0.1)
    gradtfun_seg.AddPoint(0.6, 0.3)
    gradtfun_seg.AddPoint(1.0, 0.6)

    ctfun_seg = vtk.vtkColorTransferFunction()
    ctfun_seg.AddRGBPoint(minValue, 0.1, 0.9, 0.0)
    ctfun_seg.AddRGBPoint(maxValue, 0.1, 0.9, 0.3)

    # outline = vtk.vtkOutlineFilter()
    # outline.SetInputConnection(img_arr_seg.GetOutputPort())
    # outlineMapper = vtk.vtkPolyDataMapper()
    # outlineMapper.SetInputConnection(outline.GetOutputPort())
    # outlineActor = vtk.vtkActor()
    # outlineActor.SetMapper(outlineMapper)

    volumeMapper_seg = vtk.vtkGPUVolumeRayCastMapper()
    volumeMapper_seg.SetInputConnection(img_arr_seg.GetOutputPort())

    volumeProperty_seg = vtk.vtkVolumeProperty()
    volumeProperty_seg.SetColor(ctfun_seg)
    volumeProperty_seg.SetScalarOpacity(tcfun_seg)
    volumeProperty_seg.SetGradientOpacity(gradtfun_seg)
    volumeProperty_seg.SetInterpolationTypeToLinear()
    volumeProperty_seg.ShadeOn()

    render_volume_seg = vtk.vtkVolume()
    render_volume_seg.SetMapper(volumeMapper_seg)
    render_volume_seg.SetProperty(volumeProperty_seg)

    # render.AddActor(outlineActor)
    render.AddVolume(render_volume_seg)
    render.ResetCamera()

    return render


def getRenderOfSrcImageWithClip(count, camera, renWinInteractor, numpyImage_src, spacing,
                                minValue=0, maxValue=10, pos=(0, 0, 1.0, 1.0)):
    numpyImage_src = numpyImage_src.astype(np.float32) - np.min(numpyImage_src)
    numpyImage_src = maxValue * numpyImage_src / np.max(numpyImage_src)
    print('minValue, maxValue', minValue, maxValue)

    render = vtk.vtkRenderer()
    render.SetBackground(0.8, 0.8, 0.8)
    render.SetActiveCamera(camera)
    render.SetViewport(*pos)

    img_arr = vtkImageImportFromArray()
    img_arr.SetArray(np.ascontiguousarray(numpyImage_src))
    img_arr.SetDataSpacing(spacing)
    img_arr.SetDataOrigin((0, 0, 0))
    img_arr.Update()

    tcfun = vtk.vtkPiecewiseFunction()
    tcfun.AddPoint(minValue, 0.0)
    # tcfun.AddPoint(minValue + 1, 0.3)
    tcfun.AddPoint(maxValue, 1.0)

    gradtfun = vtk.vtkPiecewiseFunction()
    gradtfun.AddPoint(0.0, 0.3)
    gradtfun.AddPoint(0.2, 0.4)
    gradtfun.AddPoint(0.6, 0.6)
    gradtfun.AddPoint(1.0, 1.0)

    ctfun = vtk.vtkColorTransferFunction()
    ctfun.AddRGBPoint(minValue, 0.0, 0.0, 0.0)
    ctfun.AddRGBPoint(maxValue, 1.0, 1.0, 1.0)

    outline = vtk.vtkOutlineFilter()
    outline.SetInputConnection(img_arr.GetOutputPort())
    outlineMapper = vtk.vtkPolyDataMapper()
    outlineMapper.SetInputConnection(outline.GetOutputPort())
    outlineActor = vtk.vtkActor()
    outlineActor.SetMapper(outlineMapper)

    dims = img_arr.GetOutput().GetDimensions()
    # print(dims)

    extractVOI = vtk.vtkExtractVOI()
    extractVOI.SetInputConnection(img_arr.GetOutputPort())
    extractVOI.SetVOI(0, dims[0] - 1, 0, dims[1] - 1, 0, dims[2] - 1)
    extractVOI.Update()

    # print(extractVOI.GetOutput().GetDimensions())

    volumeMapper_src = vtk.vtkGPUVolumeRayCastMapper()
    volumeMapper_src.SetInputData(extractVOI.GetOutput())

    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetColor(ctfun)
    volumeProperty.SetScalarOpacity(tcfun)
    volumeProperty.SetGradientOpacity(gradtfun)
    volumeProperty.SetInterpolationTypeToLinear()
    volumeProperty.ShadeOn()

    render_volume = vtk.vtkVolume()
    render_volume.SetMapper(volumeMapper_src)
    render_volume.SetProperty(volumeProperty)

    render.AddActor(outlineActor)
    render.AddVolume(render_volume)
    render.ResetCamera()

    sliderRep_min = vtk.vtkSliderRepresentation2D()
    sliderRep_min.SetMinimumValue(0)
    sliderRep_min.SetMaximumValue(10)
    sliderRep_min.SetValue(1)
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
    sliderWidget_min.SetInteractor(renWinInteractor)
    sliderWidget_min.SetRepresentation(sliderRep_min)
    sliderWidget_min.SetCurrentRenderer(render)
    sliderWidget_min.SetAnimationModeToAnimate()

    sliderRep_max = vtk.vtkSliderRepresentation2D()
    sliderRep_max.SetMinimumValue(0)
    sliderRep_max.SetMaximumValue(10)
    sliderRep_max.SetValue(9)
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
    sliderWidget_max.SetInteractor(renWinInteractor)
    sliderWidget_max.SetRepresentation(sliderRep_max)
    sliderWidget_max.SetCurrentRenderer(render)
    sliderWidget_max.SetAnimationModeToAnimate()

    def update_minmax(obj, ev):
        # print(obj)
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

        tcfun.RemoveAllPoints()
        tcfun.AddPoint(minValue, 0.0)
        tcfun.AddPoint(maxValue, 1.0)
        volumeProperty.SetScalarOpacity(tcfun)
        # print('update_minmax')

        sliceActor_i_min.GetProperty().SetColorLevel(maxValue / 2 + minValue / 2)
        sliceActor_i_min.GetProperty().SetColorWindow(maxValue - minValue)
        sliceActor_j_min.GetProperty().SetColorLevel(maxValue / 2 + minValue / 2)
        sliceActor_j_min.GetProperty().SetColorWindow(maxValue - minValue)
        sliceActor_k_min.GetProperty().SetColorLevel(maxValue / 2 + minValue / 2)
        sliceActor_k_min.GetProperty().SetColorWindow(maxValue - minValue)

        sliceActor_i_max.GetProperty().SetColorLevel(maxValue / 2 + minValue / 2)
        sliceActor_i_max.GetProperty().SetColorWindow(maxValue - minValue)
        sliceActor_j_max.GetProperty().SetColorLevel(maxValue / 2 + minValue / 2)
        sliceActor_j_max.GetProperty().SetColorWindow(maxValue - minValue)
        sliceActor_k_max.GetProperty().SetColorLevel(maxValue / 2 + minValue / 2)
        sliceActor_k_max.GetProperty().SetColorWindow(maxValue - minValue)

    ##########################################################

    sliceActor_i_min = vtk.vtkImageSlice()
    sliceMapper_i_min = vtk.vtkImageSliceMapper()
    sliceMapper_i_min.SetInputData(img_arr.GetOutput())
    sliceMapper_i_min.SetOrientationToX()
    sliceMapper_i_min.SetSliceNumber(0)
    sliceActor_i_min.SetMapper(sliceMapper_i_min)

    sliceActor_j_min = vtk.vtkImageSlice()
    sliceMapper_j_min = vtk.vtkImageSliceMapper()
    sliceMapper_j_min.SetInputData(img_arr.GetOutput())
    sliceMapper_j_min.SetOrientationToY()
    sliceMapper_j_min.SetSliceNumber(0)
    sliceActor_j_min.SetMapper(sliceMapper_j_min)

    sliceActor_k_min = vtk.vtkImageSlice()
    sliceMapper_k_min = vtk.vtkImageSliceMapper()
    sliceMapper_k_min.SetInputData(img_arr.GetOutput())
    sliceMapper_k_min.SetOrientationToZ()
    sliceMapper_k_min.SetSliceNumber(0)
    sliceActor_k_min.SetMapper(sliceMapper_k_min)

    sliceActor_i_max = vtk.vtkImageSlice()
    sliceMapper_i_max = vtk.vtkImageSliceMapper()
    sliceMapper_i_max.SetInputData(img_arr.GetOutput())
    sliceMapper_i_max.SetOrientationToX()
    sliceMapper_i_max.SetSliceNumber(dims[0])
    sliceActor_i_max.SetMapper(sliceMapper_i_max)

    sliceActor_j_max = vtk.vtkImageSlice()
    sliceMapper_j_max = vtk.vtkImageSliceMapper()
    sliceMapper_j_max.SetInputData(img_arr.GetOutput())
    sliceMapper_j_max.SetOrientationToY()
    sliceMapper_j_max.SetSliceNumber(dims[1])
    sliceActor_j_max.SetMapper(sliceMapper_j_max)

    sliceActor_k_max = vtk.vtkImageSlice()
    sliceMapper_k_max = vtk.vtkImageSliceMapper()
    sliceMapper_k_max.SetInputData(img_arr.GetOutput())
    sliceMapper_k_max.SetOrientationToZ()
    sliceMapper_k_max.SetSliceNumber(dims[2])
    sliceActor_k_max.SetMapper(sliceMapper_k_max)

    sliceActor_i_min.GetProperty().SetColorLevel(maxValue / 2 + minValue / 2)
    sliceActor_i_min.GetProperty().SetColorWindow(maxValue - minValue)
    sliceActor_j_min.GetProperty().SetColorLevel(maxValue / 2 + minValue / 2)
    sliceActor_j_min.GetProperty().SetColorWindow(maxValue - minValue)
    sliceActor_k_min.GetProperty().SetColorLevel(maxValue / 2 + minValue / 2)
    sliceActor_k_min.GetProperty().SetColorWindow(maxValue - minValue)

    sliceActor_i_max.GetProperty().SetColorLevel(maxValue / 2 + minValue / 2)
    sliceActor_i_max.GetProperty().SetColorWindow(maxValue - minValue)
    sliceActor_j_max.GetProperty().SetColorLevel(maxValue / 2 + minValue / 2)
    sliceActor_j_max.GetProperty().SetColorWindow(maxValue - minValue)
    sliceActor_k_max.GetProperty().SetColorLevel(maxValue / 2 + minValue / 2)
    sliceActor_k_max.GetProperty().SetColorWindow(maxValue - minValue)

    render.AddActor(sliceActor_i_min)
    render.AddActor(sliceActor_j_min)
    render.AddActor(sliceActor_k_min)
    render.AddActor(sliceActor_i_max)
    render.AddActor(sliceActor_j_max)
    render.AddActor(sliceActor_k_max)
    #####################################################################
    sliderWidget_min.AddObserver(vtk.vtkCommand.InteractionEvent, update_minmax)
    sliderWidget_max.AddObserver(vtk.vtkCommand.InteractionEvent, update_minmax)
    sliderWidget_min.EnabledOn()
    sliderWidget_max.EnabledOn()

    def getCropSlider(dim_index, dim_size):
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
        # print(dims)
        extractVOI.SetVOI(int(dim1_minValue), int(dim1_maxValue),
                          int(dim2_minValue), int(dim2_maxValue),
                          int(dim3_minValue), int(dim3_maxValue))
        extractVOI.Update()
        # print(extractVOI.GetOutput().GetDimensions())
        # print('update_crop')

        sliceMapper_i_min.SetSliceNumber(int(dim1_minValue))
        sliceMapper_j_min.SetSliceNumber(int(dim2_minValue))
        sliceMapper_k_min.SetSliceNumber(int(dim3_minValue))

        sliceMapper_i_max.SetSliceNumber(int(dim1_maxValue))
        sliceMapper_j_max.SetSliceNumber(int(dim2_maxValue))
        sliceMapper_k_max.SetSliceNumber(int(dim3_maxValue))

    dim1_sliderWidget_min, dim1_sliderWidget_max = getCropSlider(1, dim_size=dims[0])
    dim2_sliderWidget_min, dim2_sliderWidget_max = getCropSlider(2, dim_size=dims[1])
    dim3_sliderWidget_min, dim3_sliderWidget_max = getCropSlider(3, dim_size=dims[2])

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

    return render, sliderWidget_min, sliderWidget_max


def getRenderSrcWithSeg(camera,
                        renWinInteractor,
                        renWin,
                        numpyImage_src,
                        numpyImage_segs,
                        spacing,
                        minValue=0, maxValue=10, pos=(0, 0, 1.0, 1.0)):
    numpyImage_src = numpyImage_src.astype(np.float32) - np.min(numpyImage_src)
    numpyImage_src = maxValue * numpyImage_src / np.max(numpyImage_src)
    print('minValue, maxValue', minValue, maxValue)

    render = vtk.vtkRenderer()
    render.SetBackground(0.8, 0.8, 0.8)
    render.SetActiveCamera(camera)
    render.SetViewport(*pos)

    img_arr_src = vtkImageImportFromArray()
    img_arr_src.SetArray(np.ascontiguousarray(numpyImage_src))
    img_arr_src.SetDataSpacing(spacing)
    img_arr_src.SetDataOrigin((0, 0, 0))
    img_arr_src.Update()

    tcfun_src = vtk.vtkPiecewiseFunction()
    tcfun_src.AddPoint(minValue + 1, 1.0)
    tcfun_src.AddPoint(maxValue, 1.0)

    gradtfun_src = vtk.vtkPiecewiseFunction()
    gradtfun_src.AddPoint(0.0, 0.0)
    gradtfun_src.AddPoint(1.0, 0.6)
    gradtfun_src.AddPoint(3.0, 0.8)
    gradtfun_src.AddPoint(maxValue, 1.0)

    ctfun_src = vtk.vtkColorTransferFunction()
    ctfun_src.AddRGBPoint(minValue, 0.9, 0.1, 0.1)
    ctfun_src.AddRGBPoint(maxValue, 0.9, 0.1, 0.1)

    outline = vtk.vtkOutlineFilter()
    outline.SetInputConnection(img_arr_src.GetOutputPort())
    outlineMapper = vtk.vtkPolyDataMapper()
    outlineMapper.SetInputConnection(outline.GetOutputPort())
    outlineActor = vtk.vtkActor()
    outlineActor.SetMapper(outlineMapper)

    volumeMapper_src = vtk.vtkGPUVolumeRayCastMapper()
    volumeMapper_src.SetInputData(img_arr_src.GetOutput())

    volumeProperty_src = vtk.vtkVolumeProperty()
    volumeProperty_src.SetColor(ctfun_src)
    volumeProperty_src.SetScalarOpacity(tcfun_src)
    volumeProperty_src.SetGradientOpacity(gradtfun_src)
    volumeProperty_src.SetInterpolationTypeToLinear()
    volumeProperty_src.ShadeOn()

    render_volume_src = vtk.vtkVolume()
    render_volume_src.SetMapper(volumeMapper_src)
    render_volume_src.SetProperty(volumeProperty_src)

    render.AddActor(outlineActor)
    render.AddVolume(render_volume_src)
    volumeProperty_segs = []
    for i, numpyImage_seg in enumerate(numpyImage_segs):
        print("add seg")
        numpyImage_seg = numpyImage_seg.astype(np.float32) - np.min(numpyImage_seg)
        numpyImage_seg = maxValue * numpyImage_seg / np.max(numpyImage_seg)
        numpyImage_seg = (numpyImage_seg > 4) * 10.0

        img_arr_seg = vtkImageImportFromArray()
        img_arr_seg.SetArray(np.ascontiguousarray(numpyImage_seg))
        img_arr_seg.SetDataSpacing(spacing)
        img_arr_seg.SetDataOrigin((0, 0, 0))
        img_arr_seg.Update()

        tcfun_seg = vtk.vtkPiecewiseFunction()
        tcfun_seg.AddPoint(minValue + 1, 0.2)
        tcfun_seg.AddPoint(maxValue, 0.2)

        gradtfun_seg = vtk.vtkPiecewiseFunction()
        gradtfun_seg.AddPoint(minValue, 0.0)
        gradtfun_seg.AddPoint(1.0, 0.3)
        gradtfun_seg.AddPoint(maxValue, 0.5)

        ctfun_seg = vtk.vtkColorTransferFunction()
        ctfun_seg.AddRGBPoint(minValue, 0.9 * i, 0.9, 0.0)
        ctfun_seg.AddRGBPoint(maxValue, 0.9 * i, 0.9, 0.3)

        outline = vtk.vtkOutlineFilter()
        outline.SetInputConnection(img_arr_seg.GetOutputPort())
        outlineMapper = vtk.vtkPolyDataMapper()
        outlineMapper.SetInputConnection(outline.GetOutputPort())
        outlineActor = vtk.vtkActor()
        outlineActor.SetMapper(outlineMapper)

        volumeMapper_seg = vtk.vtkGPUVolumeRayCastMapper()
        volumeMapper_seg.SetInputData(img_arr_seg.GetOutput())

        volumeProperty_seg = vtk.vtkVolumeProperty()
        volumeProperty_seg.SetColor(ctfun_seg)
        volumeProperty_seg.SetScalarOpacity(tcfun_seg)
        volumeProperty_seg.SetGradientOpacity(gradtfun_seg)
        volumeProperty_seg.SetInterpolationTypeToLinear()
        volumeProperty_seg.ShadeOn()
        volumeProperty_segs.append(volumeProperty_seg)

        render_volume_seg = vtk.vtkVolume()
        render_volume_seg.SetMapper(volumeMapper_seg)
        render_volume_seg.SetProperty(volumeProperty_seg)

        render.AddActor(outlineActor)
        render.AddVolume(render_volume_seg)

    render.ResetCamera()

    sliderRep_min = vtk.vtkSliderRepresentation2D()
    sliderRep_min.SetMinimumValue(0)
    sliderRep_min.SetMaximumValue(10)
    sliderRep_min.SetValue(1)
    sliderRep_min.SetTitleText("minValue")
    sliderRep_min.SetSliderLength(0.025)
    sliderRep_min.SetSliderWidth(0.05)
    sliderRep_min.SetEndCapLength(0.005)
    sliderRep_min.SetEndCapWidth(0.025)
    sliderRep_min.SetTubeWidth(0.0125)
    sliderRep_min.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
    sliderRep_min.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
    sliderRep_min.GetPoint1Coordinate().SetValue(0.15 / 1, 0.1)
    sliderRep_min.GetPoint2Coordinate().SetValue(0.45 / 1, 0.1)

    sliderWidget_min = vtk.vtkSliderWidget()
    sliderWidget_min.SetInteractor(renWinInteractor)
    sliderWidget_min.SetRepresentation(sliderRep_min)
    sliderWidget_min.SetCurrentRenderer(render)
    sliderWidget_min.SetAnimationModeToAnimate()

    sliderRep_max = vtk.vtkSliderRepresentation2D()
    sliderRep_max.SetMinimumValue(0)
    sliderRep_max.SetMaximumValue(10)
    sliderRep_max.SetValue(9)
    sliderRep_max.SetTitleText("maxValue")
    sliderRep_max.SetSliderLength(0.025)
    sliderRep_max.SetSliderWidth(0.05)
    sliderRep_max.SetEndCapLength(0.005)
    sliderRep_max.SetEndCapWidth(0.025)
    sliderRep_max.SetTubeWidth(0.0125)
    sliderRep_max.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
    sliderRep_max.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
    sliderRep_max.GetPoint1Coordinate().SetValue(0.55 / 1, 0.1)
    sliderRep_max.GetPoint2Coordinate().SetValue(0.85 / 1, 0.1)

    sliderWidget_max = vtk.vtkSliderWidget()
    sliderWidget_max.SetInteractor(renWinInteractor)
    sliderWidget_max.SetRepresentation(sliderRep_max)
    sliderWidget_max.SetCurrentRenderer(render)
    sliderWidget_max.SetAnimationModeToAnimate()

    def update_minmax(obj, ev):
        # print(obj)
        minValue = sliderWidget_min.GetRepresentation().GetValue()
        maxValue = sliderWidget_max.GetRepresentation().GetValue()
        # reset value
        if minValue >= maxValue:
            if obj == sliderWidget_max:
                sliderWidget_max.GetRepresentation().SetValue(max(maxValue, minValue + 0.01))
            elif obj == sliderWidget_min:
                sliderWidget_min.GetRepresentation().SetValue(min(maxValue - 0.01, minValue))
        minValue = sliderWidget_min.GetRepresentation().GetValue()
        maxValue = sliderWidget_max.GetRepresentation().GetValue()

        tcfun_src.RemoveAllPoints()
        tcfun_src.AddPoint(minValue, 0.0)
        tcfun_src.AddPoint(maxValue, 1.0)
        volumeProperty_src.SetScalarOpacity(tcfun_src)
        # print(minValue, maxValue)

    sliderWidget_min.AddObserver(vtk.vtkCommand.InteractionEvent, update_minmax)
    sliderWidget_max.AddObserver(vtk.vtkCommand.InteractionEvent, update_minmax)
    sliderWidget_min.EnabledOn()
    sliderWidget_max.EnabledOn()

    class KeyPressInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):

        def __init__(self, parent=None, *args, **kwargs):
            super(KeyPressInteractorStyle).__init__(*args, **kwargs)
            self.parent = vtk.vtkRenderWindowInteractor()
            if parent is not None:
                self.parent = parent

            self.AddObserver("KeyPressEvent", self.keyPress)

        def keyPress(self, obj, event):
            key = self.parent.GetKeySym()
            if key.upper() == 'X':
                opacity = tcfun_seg.GetValue(0)
                if opacity:
                    print('Hide Label')
                    tcfun_seg.RemoveAllPoints()
                    tcfun_seg.AddPoint(minValue, 0.0)
                    tcfun_seg.AddPoint(maxValue, 0.0)
                    for volumeProperty_seg in volumeProperty_segs:
                        volumeProperty_seg.SetScalarOpacity(tcfun_seg)
                    renWin.Render()

                else:
                    print('Show Label')
                    tcfun_seg.RemoveAllPoints()
                    tcfun_seg.AddPoint(minValue + 1, 0.2)
                    tcfun_seg.AddPoint(maxValue, 0.2)
                    for volumeProperty_seg in volumeProperty_segs:
                        volumeProperty_seg.SetScalarOpacity(tcfun_seg)
                    renWin.Render()

            if key == 'Down':
                # print('Down')
                # tfun.RemoveAllPoints()
                # tfun.AddPoint(1129, 0)
                renWin.Render()

    renWinInteractor.SetInteractorStyle(KeyPressInteractorStyle(parent=renWinInteractor))  # 在交互操作里面添加这个自定义的操作例如up,down

    return render, sliderWidget_min, sliderWidget_max


def vtkShowTogether(numpyImage_src, numpyImage_segs, spacing=(1.0, 1.0, 1.0)):
    assert isinstance(numpyImage_src, np.ndarray)
    if isinstance(numpyImage_segs, np.ndarray):
        numpyImage_segs = [numpyImage_segs]
    assert isinstance(numpyImage_segs, (list, tuple)), "numpyImage_segs must be one of list or tuple"

    num_seg = len(numpyImage_segs)
    assert 0 <= num_seg <= 4

    spacing = tuple(reversed(spacing))

    renWin = vtk.vtkRenderWindow()
    renWinInteractor = vtk.vtkRenderWindowInteractor()
    renWinInteractor.SetRenderWindow(renWin)

    renWin.SetSize(300, 300)
    # print(col, row)
    camera = vtk.vtkCamera()

    print('shape of data', numpyImage_src.shape)
    pos = (0, 0, 1.0, 1.0)
    render, sliderWidget_min, sliderWidget_max = getRenderSrcWithSeg(camera, renWinInteractor, renWin,
                                                                     numpyImage_src, numpyImage_segs, spacing,
                                                                     pos=pos)
    # render, sliderWidget_min, sliderWidget_max = getRenderOfSrcImage(1,
    #                                                                  camera, renWinInteractor,
    #                                                                  numpyImage_src, spacing,
    #                                                                  pos=pos)
    renWin.AddRenderer(render)

    # renWinInteractor.SetInteractorStyle(KeyPressInteractorStyle(parent=renWinInteractor))  # 在交互操作里面添加这个自定义的操作例如up,down
    # renWinInteractor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())  # 在交互操作里面添加这个自定义的操作例如up,down
    renWin.Render()
    renWinInteractor.Start()


def vtkShowMulti(numpyImage_srcs, numpyImage_segs, spacing=(1.0, 1.0, 1.0)):
    if isinstance(numpyImage_srcs, np.ndarray):
        numpyImage_srcs = [numpyImage_srcs]
    if isinstance(numpyImage_segs, np.ndarray):
        numpyImage_segs = [numpyImage_segs]
    assert isinstance(numpyImage_srcs, (list, tuple)), "numpyImage_srcs must be one of list or tuple"
    assert isinstance(numpyImage_segs, (list, tuple)), "numpyImage_segs must be one of list or tuple"

    num_src = len(numpyImage_srcs)
    num_seg = len(numpyImage_segs)
    assert 0 <= num_src <= 4 and 0 <= num_seg <= 4

    spacing = tuple(reversed(spacing))

    # 键盘控制交互式操作
    class KeyPressInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):

        def __init__(self, parent=None, *args, **kwargs):
            super(KeyPressInteractorStyle).__init__(*args, **kwargs)
            self.parent = vtk.vtkRenderWindowInteractor()
            if parent is not None:
                self.parent = parent

            self.AddObserver("KeyPressEvent", self.keyPress)

        def keyPress(self, obj, event):
            key = self.parent.GetKeySym()
            if key == 'Up':
                # gradtfun.AddPoint(-100, 1.0)
                # gradtfun.AddPoint(10, 1.0)
                # gradtfun.AddPoint(20, 1.0)
                #
                # volumeProperty.SetGradientOpacity(gradtfun)
                renWin.Render()
            if key == 'Down':
                # print('Down')
                # tfun.RemoveAllPoints()
                # tfun.AddPoint(1129, 0)
                renWin.Render()

    camera = vtk.vtkCamera()

    renWin = vtk.vtkRenderWindow()
    renWinInteractor = vtk.vtkRenderWindowInteractor()
    renWinInteractor.SetRenderWindow(renWin)  # 把上面那个窗口加入交互操作

    col = max(num_seg, num_src)
    row = int(num_seg > 0) + int(num_src > 0)
    renWin.SetSize(300 * col, 300 * row)
    # print(col, row)

    for i, numpyImage_src in enumerate(numpyImage_srcs):
        pos = [i / col, 1 - 1 / row, (i + 1) / col, 1]
        print('shape of data No.', i, numpyImage_src.shape, pos)
        render, sliderWidget_min, sliderWidget_max = getRenderOfSrcImage(col,
                                                                         camera, renWinInteractor,
                                                                         numpyImage_src, spacing,
                                                                         pos=pos)
        renWin.AddRenderer(render)

    for i, numpyImage_seg in enumerate(numpyImage_segs):
        pos = [i / col, 0, (i + 1) / col, 1 / row]
        print('shape of data No.', i, numpyImage_seg.shape, pos)
        render = getRenderOfSegImage(col,
                                     camera, renWinInteractor,
                                     numpyImage_seg, spacing,
                                     pos=pos)
        renWin.AddRenderer(render)

    # renWinInteractor.SetInteractorStyle(KeyPressInteractorStyle(parent=renWinInteractor))  # 在交互操作里面添加这个自定义的操作例如up,down
    renWinInteractor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())  # 在交互操作里面添加这个自定义的操作例如up,down
    renWin.Render()
    renWinInteractor.Start()


def vtkShowNotebook(numpyImage, spacing=(1.0, 1.0, 1.0)):
    print('Running vtkShow ...')
    from importlib import import_module
    import tempfile
    import sys
    import shutil
    #
    # with tempfile.TemporaryDirectory() as temp_config_dir:
    #     print(temp_config_dir)
    #     temp_config_file = tempfile.NamedTemporaryFile(
    #         dir=temp_config_dir, suffix='.py', delete=False)
    #     temp_config_name = os.path.basename(temp_config_file.name)
    #     shutil.copyfile(os.path.dirname(__file__) + '/tmp_func.py', os.path.join(temp_config_dir, temp_config_name))
    #     temp_module_name = os.path.splitext(temp_config_name)[0]
    #     sys.path.insert(0, temp_config_dir)
    #     pickle.dump({'data': numpyImage, 'spacing': spacing}, open(temp_config_dir + '/tmp.pkl', 'wb'))
    #     mod = import_module(temp_module_name)
    #     del sys.modules[temp_module_name]
    #     # close temp file
    #     temp_config_file.close()

    # pool = multiprocessing.Pool(1)
    # pool.apply(func=_vtkShow, args=(numpyImage, spacing,))
    # pool.close()
    # pool.join()
    if os.path.exists(os.path.dirname(__file__) + '/tmp.pkl'):
        os.remove(os.path.dirname(__file__) + '/tmp.pkl')
    pickle.dump({'data': numpyImage, 'spacing': spacing}, open(os.path.dirname(__file__) + '/tmp.pkl', 'wb'))
    # print(os.path.dirname(__file__))
    cmd = f'{sys.executable} \"{os.path.dirname(__file__)}/tmp_func.py\"  --mode 1'
    print(cmd)
    os.system(cmd)
    print('closing')


def vtkShow(numpyImage, spacing=(1.0, 1.0, 1.0)):
    assert isinstance(numpyImage, np.ndarray), "numpyImage_srcs must be one of list or tuple"

    spacing = tuple(reversed(spacing))

    camera = vtk.vtkCamera()

    renWin = vtk.vtkRenderWindow()
    renWinInteractor = vtk.vtkRenderWindowInteractor()
    renWinInteractor.SetRenderWindow(renWin)  # 把上面那个窗口加入交互操作

    renWin.SetSize(450, 300)

    pos = [0, 0, 1, 1]
    print('shape of data ', numpyImage.shape, pos, spacing)
    render, sliderWidget_min, sliderWidget_max = getRenderOfSrcImage(1,
                                                                     camera, renWinInteractor,
                                                                     numpyImage, spacing,
                                                                     pos=pos)
    renWin.AddRenderer(render)

    # renWinInteractor.SetInteractorStyle(KeyPressInteractorStyle(parent=renWinInteractor))  # 在交互操作里面添加这个自定义的操作例如up,down
    renWinInteractor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())  # 在交互操作里面添加这个自定义的操作例如up,down
    renWin.Render()
    renWinInteractor.Start()
    # print('Closing')

    renWin.Finalize()
    renWinInteractor.TerminateApp()

    # renWin.Render()
    # renWin.Finalize()
    # renWinInteractor.TerminateApp()
    # renWin.End()  # will cause Notebook restart
    # del renWin, renWinInteractor
    return


def vtkScreenshot(filename, numpyImage, spacing=(1.0, 1.0, 1.0)):
    p = multiprocessing.Process(target=_vtkScreenshot, args=(filename, numpyImage, spacing,))
    p.start()
    # if os.path.exists(filename):
    #     p.terminate()  # sends a SIGTERM


def _vtkScreenshot(filename, numpyImage, spacing=(1.0, 1.0, 1.0)):
    assert isinstance(numpyImage, np.ndarray), "numpyImage_srcs must be one of list or tuple"
    d, h, w = numpyImage.shape
    spacing = tuple(reversed(spacing))

    camera = vtk.vtkCamera()
    camera.SetPosition(2 * d, 2 * h, 2 * w)

    renWin = vtk.vtkRenderWindow()
    renWin.SetSize(1024, 1024)
    renWin.SetOffScreenRendering(1)
    renWinInteractor = vtk.vtkRenderWindowInteractor()
    renWinInteractor.SetRenderWindow(renWin)  # 把上面那个窗口加入交互操作

    pos = [0, 0, 1, 1]
    print('shape of data ', numpyImage.shape, pos, spacing)
    render, sliderWidget_min, sliderWidget_max = getRenderOfSrcImage(1,
                                                                     camera, renWinInteractor,
                                                                     numpyImage, spacing,
                                                                     pos=pos)

    renWin.Render()
    renWinInteractor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())  # 在交互操作里面添加这个自定义的操作例如up,down
    renWin.AddRenderer(render)
    renWin.Render()
    # renWinInteractor.Start()

    # Screenshot
    windowToImageFilter = vtk.vtkWindowToImageFilter()
    windowToImageFilter.SetInput(renWin)
    # windowToImageFilter.Set         # set the resolution of the output image (3 times the current resolution of vtk render window)
    windowToImageFilter.SetInputBufferTypeToRGBA()  # also record the alpha (transparency) channel
    windowToImageFilter.ReadFrontBufferOff()  # read from the back buffer
    windowToImageFilter.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(filename)
    writer.SetInputConnection(windowToImageFilter.GetOutputPort())
    writer.Write()

    # renWinInteractor.SetInteractorStyle(KeyPressInteractorStyle(parent=renWinInteractor))  # 在交互操作里面添加这个自定义的操作例如up,down
    # renWin.Render()
    # renWinInteractor.Start()
    # renWin.Finalize()

    # renWinInteractor.GetRenderWindow().Finalize()
    # renWinInteractor.TerminateApp()
    # del renWin, renWinInteractor


"""
https://www.programcreek.com/python/?code=adityadua24%2Frobopy%2Frobopy-master%2Frobopy%2Fbase%2Fgraphics.py
"""
