# mmdet_tensorrt_yolox

![Detection Result](detection.jpg?raw=true "Detection Result")

## Usage

* Single Image Inference

```bash
ros2 launch mmdet_tensorrt_yolox single_image_inference.launch.xml image_path:=(image path to inference) yolox_type:=(yolox-tiny | yolox-s)
```

* Camera Demo

1. launch detection
```bash
ros2 launch mmdet_tensorrt_yolox mmdet_tensorrt_yolox.launch.xml input/image:=image_raw
```

2. launch camera
```bash
ros2 component load /detection_container usb_cam usb_cam::UsbCamNode
```

3. launch viewer
```bash
ros2 run rqt_image_view rqt_image_view
```
