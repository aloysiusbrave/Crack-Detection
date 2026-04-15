import os
import glob
import cv2
import math
from ultralytics import YOLO
from fpdf import FPDF

# ==========================================
# MODULE 1: AI INFERENCE & SEVERITY SCORING
# ==========================================
def analyze_infrastructure_batch(image_folder, model_path):
    print(f"\n[SYSTEM] Booting The Inspector's Eye...")
    print(f"[SYSTEM] Loading custom weights: {model_path}")
    model = YOLO(model_path)
    
    # Cross-platform image aggregation
    all_images = []
    valid_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    
    for ext in valid_extensions:
        search_pattern = os.path.join(image_folder, ext)
        all_images.extend(glob.glob(search_pattern))
        
    if not all_images:
        print(f"[ERROR] No valid image files found in '{image_folder}'.")
        return [], ""

    print(f"[SYSTEM] Found {len(all_images)} images. Initiating batch inference...")
    
    # Run bulk inference
    results = model.predict(source=all_images, conf=0.25, save=True)
    
    all_report_data = []
    save_directory = results[0].save_dir if results else ""
    
    print("[SYSTEM] Processing geometry and severity scoring...")
    
    # Pair original paths with YOLO results to prevent FileNotFoundError
    for orig_path, r in zip(all_images, results):
        
        # Read true image dimensions
        img = cv2.imread(orig_path)
        img_height, img_width = img.shape[:2]
        image_diagonal = math.hypot(img_width, img_height)
        
        # Path management
        yolo_filename = os.path.basename(r.path)
        annotated_image_path = os.path.join(r.save_dir, yolo_filename)
        real_filename = os.path.basename(orig_path)
        
        image_data = {
            "filename": real_filename,
            "annotated_path": annotated_image_path,
            "defects": []
        }
        
        for i, box in enumerate(r.boxes):
            coords = box.xyxy[0].tolist() 
            conf = float(box.conf[0])
            x1, y1, x2, y2 = coords
            
            # Civil Engineering Heuristic: Diagonal Crack Length
            box_w = x2 - x1
            box_h = y2 - y1
            crack_length = math.hypot(box_w, box_h)
            
            ratio = crack_length / image_diagonal
            
            # Severity Classification
            if ratio > 0.40:
                severity = "CRITICAL"
            elif ratio > 0.15:
                severity = "MAJOR"
            else:
                severity = "MINOR"
                
            image_data["defects"].append({
                "id": i + 1,
                "confidence": round(conf * 100, 2),
                "severity": severity,
                "ratio": round(ratio * 100, 2) # Stored as a percentage
            })
            
        all_report_data.append(image_data)
            
    return all_report_data, save_directory

# ==========================================
# MODULE 2: AUTOMATED PDF GENERATION
# ==========================================
def generate_pdf_report(report_data, save_dir):
    if not report_data:
        return

    print("[SYSTEM] Compiling Final PDF Report...")
    pdf = FPDF()
    
    # --- COVER PAGE ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 24)
    pdf.ln(20)
    pdf.cell(0, 20, txt="The Inspector's Eye", ln=True, align='C')
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, txt="Automated Structural Analysis Report", ln=True, align='C')
    pdf.ln(20)
    
    # Global Statistics
    total_images = len(report_data)
    total_cracks = sum(len(img["defects"]) for img in report_data)
    total_critical = sum(sum(1 for d in img["defects"] if d["severity"] == "CRITICAL") for img in report_data)
    
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, txt=f"Total Infrastructure Nodes Analyzed: {total_images}", ln=True)
    pdf.cell(0, 10, txt=f"Total Structural Defects Detected: {total_cracks}", ln=True)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(255, 0, 0) if total_critical > 0 else pdf.set_text_color(0, 128, 0)
    pdf.cell(0, 10, txt=f"Immediate Action Required (Critical Faults): {total_critical}", ln=True)
    pdf.set_text_color(0, 0, 0)
    
    # --- DETAILED ANALYSIS PAGES ---
    for img_data in report_data:
        pdf.add_page()
        
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, txt=f"Node Analysis: {img_data['filename']}", ln=True)
        pdf.ln(5)
        
        if os.path.exists(img_data['annotated_path']):
            pdf.image(img_data['annotated_path'], w=170)
        else:
            pdf.set_font("Arial", 'I', 10)
            pdf.cell(0, 10, txt="[Vision system failed to render annotated output]", ln=True)
            
        pdf.ln(10)
        
        if not img_data["defects"]:
            pdf.set_font("Arial", 'B', 12)
            pdf.set_text_color(0, 128, 0)
            pdf.cell(0, 10, txt="STATUS: NO STRUCTURAL DEFECTS DETECTED.", ln=True)
            pdf.set_text_color(0, 0, 0)
        else:
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(30, 10, "Defect ID", border=1, align='C')
            pdf.cell(50, 10, "Severity", border=1, align='C')
            pdf.cell(50, 10, "Span Ratio (%)", border=1, align='C')
            pdf.cell(50, 10, "AI Confidence (%)", border=1, align='C')
            pdf.ln()
            
            pdf.set_font("Arial", '', 10)
            for d in img_data["defects"]:
                pdf.cell(30, 10, str(d["id"]), border=1, align='C')
                pdf.cell(50, 10, d["severity"], border=1, align='C')
                pdf.cell(50, 10, str(d["ratio"]), border=1, align='C')
                pdf.cell(50, 10, str(d["confidence"]), border=1, align='C')
                pdf.ln()

    # --- FINAL EXPORT ---
    output_pdf_path = os.path.join(save_dir, "Final_Inspection_Report.pdf")
    pdf.output(output_pdf_path)
    print(f"\n==========================================")
    print(f"[SUCCESS] Report saved to: {output_pdf_path}")
    print(f"==========================================")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == '__main__':
    # 1. Paths (Update these if your folders change)
    MODEL_WEIGHTS = 'runs/detect/train9/weights/best.pt' 
    TARGET_DIRECTORY = 'test/images' 
    
    # 2. Execute the Pipeline
    extracted_data, output_directory = analyze_infrastructure_batch(TARGET_DIRECTORY, MODEL_WEIGHTS)
    
    # 3. Generate the Report
    if extracted_data:
        generate_pdf_report(extracted_data, output_directory)