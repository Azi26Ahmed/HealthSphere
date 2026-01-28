import random
import csv
import os
from fpdf import FPDF

def create_patient_pdfs():
    # Get user inputs with better prompts
    print("\n" + "="*50)
    print("PDF Report Generator".center(50))
    print("="*50 + "\n")
    
    csv_path = input("Enter full path to your CSV dataset: ").strip()
    base_name = input("Enter Disease name (e.g., stroke): ").strip().lower()
    num_pdfs = input("Number of reports to generate: ").strip()

    # Validate numerical input
    if not num_pdfs.isdigit():
        print(" Error: Please enter a valid number for reports to generate.")
        return
    num_pdfs = int(num_pdfs)

    # Read CSV data
    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            headers = next(reader, None)  # Read column headers
            
            if headers is None:
                print("Error: CSV file is empty or missing headers.")
                return
            
            all_rows = list(reader)
            
            if len(all_rows) < num_pdfs:
                print(f"Error: Only {len(all_rows)} patients available. Can't create {num_pdfs} reports.")
                return
                
            # Select random patient records
            selected_patients = random.sample(all_rows, num_pdfs)

            # Create output directory
            output_dir = os.path.join("reports", f"{base_name}")
            os.makedirs(output_dir, exist_ok=True)

            # Create PDFs
            for idx, patient_data in enumerate(selected_patients, start=1):
                pdf = FPDF()
                pdf.add_page()

                # Configure layout
                pdf.set_font("Arial", size=12)
                line_height = 8
                page_width = pdf.w - 2 * pdf.l_margin

                # Create title
                pdf.set_font("Arial", "B", 16)
                title = f"{base_name.capitalize()} Health Report #{idx}"
                pdf.cell(0, line_height + 6, title, ln=1, align='C')
                pdf.ln(10)

                # Add section header
                pdf.set_font("Arial", "B", 14)
                pdf.cell(0, line_height + 4, "Patient Health Metrics", ln=1)
                pdf.ln(8)

                # Add patient data with improved inline formatting
                pdf.set_font("Arial", size=12)
                max_header_width = max(pdf.get_string_width(header) for header in headers) + 10
                
                for header, value in zip(headers, patient_data):
                    # Print header with colon
                    pdf.set_font("Arial", 'B', 12)
                    header_text = f"{header}:"
                    pdf.cell(max_header_width, line_height, header_text)
                    
                    # Print value on same line with regular font
                    pdf.set_font("Arial", '', 12)
                    pdf.cell(0, line_height, str(value), ln=1)
                    
                    # Add spacing between items
                    pdf.ln(4) 
                    # Save PDF
                pdf_filename = os.path.join(output_dir, f"{base_name}_{idx}.pdf")
                pdf.output(pdf_filename)
                print(f"✓ Created report {idx}: {pdf_filename}")

            print(f"\n{'='*50}")
            print(f"Success! Generated {num_pdfs} reports in:")
            print(f" {os.path.abspath(output_dir)}")
            print("="*50)

    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    create_patient_pdfs()