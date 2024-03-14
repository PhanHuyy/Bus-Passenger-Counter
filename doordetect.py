from pathlib import Path
import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.point import Point
from ultralytics import YOLO
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors

current_region = None
counting_regions = [{
    "name": "Out",
    "polygon": Polygon([(180, 400), (550, 400), (550, 650), (180, 650)]),  # Polygon points
    "counts": 0,
    "dragging": False,
    "region_color": (37, 255, 225),  # BGR Value
    "text_color": (0, 0, 0),
},  # Region Text
    {
        "name": "In",
        "polygon": Polygon([(600, 0), (750, 0), (750, 800), (600, 800)]),  # Polygon points
        "counts": 0,
        "dragging": False,
        "region_color": (37, 255, 225),  # BGR Value
        "text_color": (0, 0, 0),  # Region Text Color
    }]


# Polygon([(312,388),(289,390),(474,469),(497,462)])
# Polygon([(279,392),(250,397),(423,477),(454,469)])
def mouse_callback(event, x, y, flags, param):
    global current_region
    # Mouse left button down event
    if event == cv2.EVENT_LBUTTONDOWN:
        for region in counting_regions:
            if region["polygon"].contains(Point((x, y))):
                current_region = region
                current_region["dragging"] = True
                current_region["offset_x"] = x
                current_region["offset_y"] = y

    # Mouse move event
    elif event == cv2.EVENT_MOUSEMOVE:
        if current_region is not None and current_region["dragging"]:
            dx = x - current_region["offset_x"]
            dy = y - current_region["offset_y"]
            current_region["polygon"] = Polygon(
                [(p[0] + dx, p[1] + dy) for p in current_region["polygon"].exterior.coords]
            )
            current_region["offset_x"] = x
            current_region["offset_y"] = y

    # Mouse left button up event
    elif event == cv2.EVENT_LBUTTONUP:
        if current_region is not None and current_region["dragging"]:
            current_region["dragging"] = False

combo = 0
def run2(
        weights="yolov8n.pt",
        source=input,
        device="cpu",
        classes=0,
        view_img=True,
        save_video=True,
        line_thickness=2,
        region_thickness=2,
        resolution=combo
):
    """
    Run Region counting on a video using YOLOv8 and ByteTrack.

    Supports movable region for real time counting inside specific area.
    Supports multiple regions counting.
    Regions can be Polygons or rectangle in shape

    Args:
        weights (str): Model weights path.
        source (str): Video file path.
        device (str): processing device cpu, 0, 1
        classes (list): classes to detect and track
        view_img (bool): Show results.
        save_video (bool): Save results.
        line_thickness (int): Bounding box thickness.
        track_thickness (int): Tracking line thickness
        region_thickness (int): Region thickness.
    """
    vid_frame_count = 0

    # Setup Model
    model = YOLO(f"{weights}")  # load models
    model.to("cuda") if device == "0" else model.to("cpu")  # load GPU if not move to CPU to calculate
    # Extract classes names
    names = model.model.names
    # Video setup
    videocapture = cv2.VideoCapture(source)
    frame_width, frame_height = 800, 800
    fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*"mp4v")

    # Output setup
    save_dir = increment_path(Path("ultralytics_rc_output") / "exp")  # increment file exp ->exp1 if exist_ok=false
    save_dir.mkdir(parents=True, exist_ok=True)
    # video_writer = cv2.VideoWriter(str(save_dir / f"{Path(source).stem}.mp4"), fourcc, fps, (frame_width, frame_height))
    video_writer = cv2.VideoWriter(str(save_dir / "bus.mp4"), fourcc, fps, (frame_width, frame_height))
    entering_dict = {}
    exiting_dict = {}
    entering = set()
    exiting = set()
    while videocapture.isOpened():
        success, frame = videocapture.read()
        if resolution != "None":
            try:
                frame = cv2.resize(frame, (int(resolution[:4]), int(resolution[-4:])), interpolation=cv2.INTER_CUBIC)
            except:
                frame = frame
        vid_frame_count += 1

        # Extract the results
        results = model.track(frame, persist=True, classes=classes)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            clss = results[0].boxes.cls.cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            annotator = Annotator(frame, line_width=line_thickness, example=str(names))

            for box, track_id, cls in zip(boxes, track_ids, clss):
                annotator.box_label(box, str(names[cls]), color=colors(cls, True))
                bbox_right = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2  # Bbox center
                # Check if detection inside region
                # box[2],box[3]

                if counting_regions[1]["polygon"].contains(Point((bbox_right[0], bbox_right[1]))):
                    entering_dict[track_id] = (bbox_right[0], bbox_right[1])
                if track_id in entering_dict:
                    in_ob1 = counting_regions[0]["polygon"].contains(Point((bbox_right[0], bbox_right[1])))
                    if in_ob1:
                        entering.add(track_id)
                        counting_regions[0]["counts"] = len(entering)

                if counting_regions[0]["polygon"].contains(Point((bbox_right[0], bbox_right[1]))):
                    exiting_dict[track_id] = (bbox_right[0], bbox_right[1])
                if track_id in exiting_dict:
                    out_ob1 = counting_regions[1]["polygon"].contains(Point((bbox_right[0], bbox_right[1])))
                    if out_ob1:
                        exiting.add(track_id)
                        counting_regions[1]["counts"] = len(exiting)

        # Draw regions (Polygons/Rectangles)
        for region in counting_regions:
            region_label = str(region["counts"])
            region_color = region["region_color"]
            region_text_color = region["text_color"]

            polygon_coords = np.array(region["polygon"].exterior.coords, dtype=np.int32)
            centroid_x, centroid_y = int(region["polygon"].centroid.x), int(region["polygon"].centroid.y)

            text_size, _ = cv2.getTextSize(
                region_label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=line_thickness
            )
            text_x = centroid_x - text_size[0] // 2
            text_y = centroid_y + text_size[1] // 2
            cv2.rectangle(
                frame,
                (text_x - 5, text_y - text_size[1] - 5),
                (text_x + text_size[0] + 5, text_y + 5),
                region_color,
                -1,
            )
            cv2.putText(
                frame, f"{region['name']}:{region_label}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                region_text_color, line_thickness
            )
            cv2.polylines(frame, [polygon_coords], isClosed=True, color=region_color, thickness=region_thickness)

        if view_img:
            if vid_frame_count == 1:
                cv2.namedWindow("Ultralytics YOLOv8 Region Counter Movable")
                cv2.setMouseCallback("Ultralytics YOLOv8 Region Counter Movable", mouse_callback)
            cv2.imshow("Ultralytics YOLOv8 Region Counter Movable", frame)

        if save_video:
            video_writer.write(frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    del vid_frame_count
    video_writer.release()
    videocapture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run2()  # run function