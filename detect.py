import sys
import fitz
import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QApplication, QLabel, QPushButton, QVBoxLayout, QWidget,
    QFileDialog, QScrollArea, QHBoxLayout, QListWidget, QListWidgetItem
)
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QImage
from PyQt5.QtCore import QRect, Qt
from collections import defaultdict, deque

class PageRegionWidget(QLabel):
    def __init__(self, pix, regions, shared_regions, cell_w, cell_h, split_columns=None):
        super().__init__()
        self.pix = pix
        self.regions = regions
        self.shared_regions = shared_regions
        self.cell_w = cell_w
        self.cell_h = cell_h
        self.split_columns = split_columns or []
        self.display()

    def display(self):
        image = QImage(self.pix.samples, self.pix.width, self.pix.height, self.pix.stride, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        painter = QPainter(pixmap)
        # Draw all red blocks (regions)
        for rect in self.regions:
            painter.setPen(QPen(QColor(255, 0, 0), 1))
            painter.drawRect(rect)
        # Draw all green blocks (shared_regions)
        for (r, c) in self.shared_regions:
            x0, y0 = c * self.cell_w, r * self.cell_h
            painter.setPen(QPen(QColor(0, 200, 0), 2))  # green
            painter.drawRect(QRect(x0, y0, self.cell_w, self.cell_h))

        # --- Start new full-page y-axis scanning logic for orange rectangles ---
        from collections import defaultdict

        # Identify and exclude central green columns with >=80 vertical blocks
        column_rows = defaultdict(set)
        for (r, c) in self.shared_regions:
            column_rows[c].add(r)
        exclude_columns = {c for c, rows in column_rows.items() if len(rows) >= 80}

        # Draw vertical black rectangles for excluded columns
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 100))  # Semi-transparent black
        for c in exclude_columns:
            x = c * self.cell_w
            painter.drawRect(QRect(x, 0, self.cell_w, self.pix.height))

        # Define left and right ranges based on exclude_columns
        if exclude_columns:
            min_exclude_col = min(exclude_columns)
            max_exclude_col = max(exclude_columns)
        else:
            min_exclude_col = self.pix.width // self.cell_w // 2
            max_exclude_col = min_exclude_col

        # Draw left and right regions
        min_exclude_x = min_exclude_col * self.cell_w
        max_exclude_x = (max_exclude_col + 1) * self.cell_w

        painter.setPen(QPen(QColor(255, 105, 180, 120), 2, Qt.DashLine))  # Pink for left
        painter.drawRect(QRect(0, 0, min_exclude_x, self.pix.height))
        painter.setPen(QPen(QColor(0, 255, 255, 120), 2, Qt.DashLine))  # Cyan for right
        painter.drawRect(QRect(max_exclude_x, 0, self.pix.width - max_exclude_x, self.pix.height))

        # For each y-row from 0 to 200, check green block coverage in left and right ranges
        max_rows = 200
        green_blocks_by_row = defaultdict(set)
        for (r, c) in self.shared_regions:
            green_blocks_by_row[r].add(c)

        # Determine cutoff rows for left and right sides based on >60% green coverage
        cutoff_row_left = None
        cutoff_row_right = None

        left_range_cols = set(range(0, min_exclude_col))
        right_range_cols = set(range(max_exclude_col + 1, self.pix.width // self.cell_w))

        for row in range(max_rows):
            green_cols = green_blocks_by_row.get(row, set())
            # Left side coverage
            left_green = green_cols.intersection(left_range_cols)
            if left_range_cols:
                left_ratio = len(left_green) / len(left_range_cols)
                if left_ratio > 0.60:
                    cutoff_row_left = row
                    break

        for row in range(max_rows):
            green_cols = green_blocks_by_row.get(row, set())
            # Right side coverage
            right_green = green_cols.intersection(right_range_cols)
            if right_range_cols:
                right_ratio = len(right_green) / len(right_range_cols)
                if right_ratio > 0.60:
                    cutoff_row_right = row
                    break

        # Filter red blocks below cutoff rows for left and right
        def is_left_block(block):
            r, c = block
            return c in left_range_cols and (cutoff_row_left is None or r >= cutoff_row_left)

        def is_right_block(block):
            r, c = block
            return c in right_range_cols and (cutoff_row_right is None or r >= cutoff_row_right)

        red_blocks = []
        for rect in self.regions:
            r = rect.y() // self.cell_h
            c = rect.x() // self.cell_w
            red_blocks.append((r, c))

        left_blocks = [b for b in red_blocks if is_left_block(b)]
        right_blocks = [b for b in red_blocks if is_right_block(b)]

        # Visualize detected row lines across the page for red blocks below cutoff rows
        painter.setPen(QPen(QColor(100, 100, 255, 100), 1, Qt.DotLine))
        for (r, c) in red_blocks:
            if (cutoff_row_left is not None and r >= cutoff_row_left) or (cutoff_row_right is not None and r >= cutoff_row_right):
                y = r * self.cell_h
                painter.drawLine(0, y, self.pix.width, y)

        def group_by_vertical_gap(blocks):
            if not blocks:
                return []

            blocks.sort(key=lambda b: b[0])  # sort by row index
            groups = []
            current = [blocks[0]]

            for b in blocks[1:]:
                if b[0] - current[-1][0] >= 30:
                    groups.append(current)
                    current = []
                current.append(b)

            if current:
                groups.append(current)

            return groups[:2]

        self.left_groups = group_by_vertical_gap(left_blocks)
        self.right_groups = group_by_vertical_gap(right_blocks)

        # Filter out yellow regions with topmost row (min r) ≤ 50 (bottom = r=0)
        def filter_by_y_threshold(groups, threshold=150):
            filtered = []
            for group in groups:
                min_r = min(r for r, c in group)
                if min_r <= threshold:
                    filtered.append(group)
            return filtered

        self.left_groups = filter_by_y_threshold(self.left_groups)
        self.right_groups = filter_by_y_threshold(self.right_groups)

        print(f"LEFT GROUP COUNT: {len(self.left_groups)}")
        for i, group in enumerate(self.left_groups):
            rows = sorted(set(r for r, c in group))
            print(f"LEFT GROUP {i+1}: rows {rows}")

        print(f"RIGHT GROUP COUNT: {len(self.right_groups)}")
        for i, group in enumerate(self.right_groups):
            rows = sorted(set(r for r, c in group))
            print(f"RIGHT GROUP {i+1}: rows {rows}")

        painter.setPen(QPen(QColor(255, 255, 0), 3))  # Yellow
        for group in self.left_groups:
            if not group:
                continue
            x0 = min(c for r, c in group) * self.cell_w
            y0 = min(r for r, c in group) * self.cell_h
            x1 = (max(c for r, c in group) + 1) * self.cell_w
            y1 = (max(r for r, c in group) + 1) * self.cell_h
            painter.drawRect(QRect(x0, y0, x1 - x0, y1 - y0))

        for group in self.right_groups:
            if not group:
                continue
            x0 = min(c for r, c in group) * self.cell_w
            y0 = min(r for r, c in group) * self.cell_h
            x1 = (max(c for r, c in group) + 1) * self.cell_w
            y1 = (max(r for r, c in group) + 1) * self.cell_h
            painter.drawRect(QRect(x0, y0, x1 - x0, y1 - y0))
        # --- End full-page y-axis scanning logic ---

        painter.setPen(QPen(QColor(0, 255, 0, 150), 2))  # Green
        for group in self.left_groups + self.right_groups:
            if not group:
                continue
            x0 = min(c for r, c in group) * self.cell_w
            y0 = min(r for r, c in group) * self.cell_h
            x1 = (max(c for r, c in group) + 1) * self.cell_w
            y1 = (max(r for r, c in group) + 1) * self.cell_h
            painter.drawRect(QRect(x0, y0, x1 - x0, y1 - y0))

        painter.end()
        scaled = pixmap.scaled(700, 1000, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)

    def save_yellow_regions(self, page_index):
        yellow_groups = self.left_groups + self.right_groups
        for idx, group in enumerate(yellow_groups):
            if not group:
                continue
            x0 = min(c for r, c in group) * self.cell_w
            y0 = min(r for r, c in group) * self.cell_h
            x1 = (max(c for r, c in group) + 1) * self.cell_w
            y1 = (max(r for r, c in group) + 1) * self.cell_h
            self.yellow_regions.append((x0, y0, x1, y1, page_index, idx))

class SimplePDFRegionChecker(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PDF Multi-Page Region Detector")
        self.pages = []
        self.cell_w = 0
        self.cell_h = 0
        self.region_presence_counter = {}

        self.load_button = QPushButton("📄 Load PDF")
        self.load_button.clicked.connect(self.load_pdf)

        # Remove old save_button, add new save buttons
        self.save_all_button = QPushButton("📁 전체 이미지 저장")
        self.save_all_button.clicked.connect(self.save_all_yellow_regions)
        self.save_selected_button = QPushButton("💾 선택한 이미지 저장")
        self.save_selected_button.clicked.connect(self.save_selected_yellow_regions)

        # Left: scroll area for PDF pages
        self.scroll_area = QScrollArea()
        self.page_container = QWidget()
        self.page_layout = QVBoxLayout()
        self.page_container.setLayout(self.page_layout)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.page_container)

        # Right: list widget for PDF file names
        self.file_list_widget = QListWidget()
        self.file_list_widget.itemClicked.connect(self.on_file_selected)

        # Arrange layout: buttons at top, then horizontal split (left viewer, right list)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.save_all_button)
        button_layout.addWidget(self.save_selected_button)

        main_layout = QVBoxLayout()
        main_layout.addLayout(button_layout)
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.scroll_area, stretch=4)
        h_layout.addWidget(self.file_list_widget, stretch=1)
        main_layout.addLayout(h_layout)
        self.setLayout(main_layout)

        # Internal state for selected file
        self.file_path_map = {}  # file_path -> fitz.Document
        self.widgets_by_file = {}  # file_path -> list of PageRegionWidget
        self.current_file_path = None

    def load_pdf(self):
        # Allow multiple PDF selection
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Open PDFs", "", "PDF Files (*.pdf)")
        if not file_paths:
            return
        self.file_path_map = {}
        self.widgets_by_file = {}
        self.pages.clear()
        self.region_presence_counter = {}
        self.cell_w = 0
        self.cell_h = 0
        self.current_file_path = None
        # Clear the list widget and viewer area
        self.file_list_widget.clear()
        while self.page_layout.count():
            item = self.page_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        # Prepare and store all widgets with file association, but do not add to layout yet
        rows, cols = 200, 200
        for file_path in file_paths:
            doc = fitz.open(file_path)
            self.file_path_map[file_path] = doc
            # We'll store tuples of (pix, regions) for each page of this doc
            file_pages = []
            dark_cells = set()
            for i in range(len(doc)):
                page = doc.load_page(i)
                pix = page.get_pixmap(dpi=400)
                w, h = pix.width, pix.height
                cell_w = w // cols
                cell_h = h // rows
                if self.cell_w == 0:
                    self.cell_w, self.cell_h = cell_w, cell_h
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                # --- Enhanced denoising and binarization pipeline ---
                # Apply CLAHE to enhance local contrast
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                clahe_applied = clahe.apply(gray)

                # Adaptive thresholding to binarize
                binary = cv2.adaptiveThreshold(
                    clahe_applied, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    blockSize=11, C=8
                )

                # Invert and apply morphological opening to remove small noise
                inv = 255 - binary
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                opened = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel, iterations=1)

                # Final image for region detection
                gray = 255 - opened
                regions = []
                for r in range(rows):
                    for c in range(cols):
                        x0, y0 = c * cell_w, r * cell_h
                        x1, y1 = x0 + cell_w, y0 + cell_h
                        crop = gray[y0:y1, x0:x1]
                        white_ratio = np.sum(crop > 245) / crop.size
                        if white_ratio < 0.9:
                            regions.append(QRect(x0, y0, cell_w, cell_h))
                            dark_cells.add((i, r, c))  # page index included
                file_pages.append((pix, regions))
            # Clustering on dark_cells for this file
            neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            def bfs(start, region_set, visited):
                queue = deque([start])
                group = set([start])
                visited.add(start)
                while queue:
                    idx, r, c = queue.popleft()
                    for dr, dc in neighbors:
                        nr, nc = r + dr, c + dc
                        neighbor = (idx, nr, nc)
                        if neighbor in region_set and neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                            group.add(neighbor)
                return group
            region_set = set(dark_cells)
            visited = set()
            green_clusters = set()
            while region_set:
                current = region_set.pop()
                group = bfs(current, region_set, visited)
                if len(group) >= 120:
                    green_clusters.update(group)
                region_set -= group
            page_clusters_by_column = defaultdict(set)
            for (i, r, c) in green_clusters:
                page_clusters_by_column[(i, c)].add(r)
            split_columns_by_page = defaultdict(list)
            for (i, c), rows_set in page_clusters_by_column.items():
                if len(rows_set) >= 150:
                    split_columns_by_page[i].append(c)
            # For each page in this file, create widget and associate with file_path
            widgets = []
            for page_idx, (pix, regions) in enumerate(file_pages):
                green_rects = set((r, c) for (i2, r, c) in green_clusters if i2 == page_idx)
                split_cols = split_columns_by_page.get(page_idx, [])
                widget = PageRegionWidget(pix, regions, green_rects, self.cell_w, self.cell_h, split_cols)
                widgets.append(widget)
            self.widgets_by_file[file_path] = widgets
            # Add file name to right-side list widget
            item = QListWidgetItem(file_path.split("/")[-1])
            item.setData(Qt.UserRole, file_path)
            self.file_list_widget.addItem(item)
        # If any file loaded, select the first one
        if file_paths:
            self.file_list_widget.setCurrentRow(0)
            self.display_pdf_pages(file_paths[0])

    def display_pdf_pages(self, file_path):
        # Clean existing widgets safely
        if self.current_file_path:
            old_widgets = self.widgets_by_file.get(self.current_file_path, [])
            for w in old_widgets:
                w.setParent(None)
                w.deleteLater()

        while self.page_layout.count():
            item = self.page_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
                item.widget().deleteLater()
        widgets = self.widgets_by_file.get(file_path, [])
        for widget in widgets:
            self.page_layout.addWidget(widget)
        self.page_container.adjustSize()
        self.scroll_area.update()
        self.scroll_area.repaint()
        self.current_file_path = file_path

    def on_file_selected(self, item):
        file_path = item.data(Qt.UserRole)
        if file_path != self.current_file_path:
            self.display_pdf_pages(file_path)

    def save_all_yellow_regions(self):
        import os
        os.makedirs("saved_regions", exist_ok=True)
        SCALE = 72 / 400  # Scale coordinates from 400 DPI image space to 72 DPI PDF space
        # For each PDF file loaded
        for file_path, doc in self.file_path_map.items():
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_dir = os.path.join("saved_regions", base_name)
            os.makedirs(output_dir, exist_ok=True)
            widgets = self.widgets_by_file[file_path]
            for page_index, widget in enumerate(widgets):
                if widget.__class__.__name__ != "PageRegionWidget":
                    continue
                widget.yellow_regions = []
                widget.save_yellow_regions(page_index)
                page = doc.load_page(page_index)
                for (x0, y0, x1, y1, page_index2, region_idx) in widget.yellow_regions:
                    x0, x1 = sorted([int(x0), int(x1)])
                    y0, y1 = sorted([int(y0), int(y1)])
                    if x1 - x0 < 10:
                        x1 = x0 + 10
                    if y1 - y0 < 10:
                        y1 = y0 + 10
                    print(f"Saving region {region_idx+1} of page {page_index}: ({x0}, {y0}, {x1}, {y1}) in {output_dir}")
                    try:
                        clipped_rect = fitz.Rect(x0 * SCALE, y0 * SCALE, x1 * SCALE, y1 * SCALE)
                        clipped_pix = page.get_pixmap(clip=clipped_rect, dpi=400)
                        output_path = os.path.join(output_dir, f"page_{page_index}_region_{region_idx+1}.png")
                        clipped_pix.save(output_path)
                    except Exception as e:
                        print(f"[ERROR] Failed to save region {region_idx+1} of page {page_index}: {e}")

    def save_selected_yellow_regions(self):
        import os
        os.makedirs("saved_regions", exist_ok=True)
        SCALE = 72 / 400  # Scale coordinates from 400 DPI image space to 72 DPI PDF space
        file_path = self.current_file_path
        if not file_path or file_path not in self.file_path_map:
            print("[ERROR] No PDF selected for saving.")
            return
        doc = self.file_path_map[file_path]
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_dir = os.path.join("saved_regions", base_name)
        os.makedirs(output_dir, exist_ok=True)
        widgets = self.widgets_by_file[file_path]
        for page_index, widget in enumerate(widgets):
            if widget.__class__.__name__ != "PageRegionWidget":
                continue
            widget.yellow_regions = []
            widget.save_yellow_regions(page_index)
            page = doc.load_page(page_index)
            for (x0, y0, x1, y1, page_index2, region_idx) in widget.yellow_regions:
                x0, x1 = sorted([int(x0), int(x1)])
                y0, y1 = sorted([int(y0), int(y1)])
                if x1 - x0 < 10:
                    x1 = x0 + 10
                if y1 - y0 < 10:
                    y1 = y0 + 10
                print(f"Saving region {region_idx+1} of page {page_index}: ({x0}, {y0}, {x1}, {y1}) in {output_dir}")
                try:
                    clipped_rect = fitz.Rect(x0 * SCALE, y0 * SCALE, x1 * SCALE, y1 * SCALE)
                    clipped_pix = page.get_pixmap(clip=clipped_rect, dpi=400)
                    output_path = os.path.join(output_dir, f"page_{page_index}_region_{region_idx+1}.png")
                    clipped_pix.save(output_path)
                except Exception as e:
                    print(f"[ERROR] Failed to save region {region_idx+1} of page {page_index}: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SimplePDFRegionChecker()
    window.resize(900, 1200)
    window.show()
    sys.exit(app.exec_())