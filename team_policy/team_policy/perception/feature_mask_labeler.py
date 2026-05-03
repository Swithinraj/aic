import argparse
import json
from pathlib import Path
import tkinter as tk
from tkinter import simpledialog

import cv2
import numpy as np


class TaskboardFeatureAnnotator:
    def __init__(self, image_path: str, output_json: str | None = None):
        self.image_path = Path(image_path)
        self.output_json = Path(output_json) if output_json else self.image_path.with_suffix(".features.json")

        self.image = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        if self.image is None:
            raise FileNotFoundError(f"Could not load image: {self.image_path}")

        self.base_image = self.image.copy()
        self.display_image = self.image.copy()
        self.annotations = []
        self.dragging = False
        self.start_pt = None
        self.end_pt = None
        self.selected_idx = -1

        self.window_name = "Taskboard Feature Annotator"
        self.root = tk.Tk()
        self.root.withdraw()

        if self.output_json.exists():
            self._load_annotations()

    def _load_annotations(self):
        data = json.loads(self.output_json.read_text())
        self.annotations = data.get("annotations", [])

    def _save_annotations(self):
        data = {
            "image": str(self.image_path),
            "width": int(self.image.shape[1]),
            "height": int(self.image.shape[0]),
            "annotations": self.annotations,
        }
        self.output_json.write_text(json.dumps(data, indent=2))
        print(f"Saved: {self.output_json}")

    def _draw(self):
        canvas = self.base_image.copy()

        for i, ann in enumerate(self.annotations):
            x1, y1, x2, y2 = ann["bbox_xyxy"]
            color = (0, 255, 0) if i != self.selected_idx else (0, 255, 255)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = ann["label"]
            text_bg_y1 = max(0, y1 - 24)
            cv2.rectangle(canvas, (x1, text_bg_y1), (x1 + max(120, 10 * len(label)), y1), color, -1)
            cv2.putText(canvas, label, (x1 + 4, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

        if self.dragging and self.start_pt and self.end_pt:
            cv2.rectangle(canvas, self.start_pt, self.end_pt, (255, 0, 0), 2)

        self.display_image = canvas
        cv2.imshow(self.window_name, self.display_image)

    def _normalize_box(self, p1, p2):
        x1 = min(p1[0], p2[0])
        y1 = min(p1[1], p2[1])
        x2 = max(p1[0], p2[0])
        y2 = max(p1[1], p2[1])
        return x1, y1, x2, y2

    def _find_box_at_point(self, x, y):
        for i in range(len(self.annotations) - 1, -1, -1):
            x1, y1, x2, y2 = self.annotations[i]["bbox_xyxy"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                return i
        return -1

    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            idx = self._find_box_at_point(x, y)
            if idx >= 0:
                self.selected_idx = idx
                self._draw()
                return
            self.dragging = True
            self.start_pt = (x, y)
            self.end_pt = (x, y)
            self.selected_idx = -1
            self._draw()

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.end_pt = (x, y)
            self._draw()

        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            self.end_pt = (x, y)

            x1, y1, x2, y2 = self._normalize_box(self.start_pt, self.end_pt)
            if (x2 - x1) < 5 or (y2 - y1) < 5:
                self.start_pt = None
                self.end_pt = None
                self._draw()
                return

            label = simpledialog.askstring("Feature Name", "Enter feature name:")
            if label is not None:
                label = label.strip()
                if label:
                    ann = {
                        "label": label,
                        "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                    }
                    self.annotations.append(ann)
                    self.selected_idx = len(self.annotations) - 1

            self.start_pt = None
            self.end_pt = None
            self._draw()

        elif event == cv2.EVENT_RBUTTONDOWN:
            idx = self._find_box_at_point(x, y)
            if idx >= 0:
                self.selected_idx = idx
                current = self.annotations[idx]["label"]
                new_label = simpledialog.askstring("Rename Feature", "Edit feature name:", initialvalue=current)
                if new_label is not None:
                    new_label = new_label.strip()
                    if new_label:
                        self.annotations[idx]["label"] = new_label
                self._draw()

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, min(1600, self.image.shape[1]), min(1000, self.image.shape[0]))
        cv2.setMouseCallback(self.window_name, self._mouse_cb)
        self._draw()

        print("Controls:")
        print("Left drag: draw box")
        print("Right click on box: rename")
        print("Delete or Backspace: delete selected box")
        print("u: undo last")
        print("c: clear all")
        print("s: save")
        print("q or Esc: save and quit")

        while True:
            key = cv2.waitKey(20) & 0xFF

            if key in (27, ord("q")):
                self._save_annotations()
                break
            elif key == ord("s"):
                self._save_annotations()
            elif key == ord("u"):
                if self.annotations:
                    self.annotations.pop()
                    self.selected_idx = -1
                    self._draw()
            elif key == ord("c"):
                self.annotations = []
                self.selected_idx = -1
                self._draw()
            elif key in (8, 127):
                if 0 <= self.selected_idx < len(self.annotations):
                    self.annotations.pop(self.selected_idx)
                    self.selected_idx = -1
                    self._draw()

        cv2.destroyAllWindows()
        self.root.destroy()


def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("image", help="Path to PNG image")
        parser.add_argument("--output", default=None, help="Output JSON path")
        args = parser.parse_args()

        tool = TaskboardFeatureAnnotator(args.image, args.output)
        tool.run()


if __name__ == "__main__":
    main()