import easyocr  # please install this lib
import argparse
import os
import openslide
import numpy as np
import re


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='path that contains WSI files (will search recursively)', default='/jhcnas5/jmabq/Pathology/NanfangHospital/WSIs/')
    parser.add_argument('--format', help='data format', default='svs')
    parser.add_argument('--hospital', help='hospital name', default='Hebeisiyuan')
    parser.add_argument('--recursive', help='search recursively in subdirectories', action='store_true', default=True)
    parser.add_argument('--output', help='path to save the output CSV file', default=None)
    args = parser.parse_args()
    return args


def find_wsi_files(root_path, format_extension, recursive=True):
    """Find all WSI files with the specified format in the directory and its subdirectories."""
    wsi_files = []
    
    if os.path.isfile(root_path) and format_extension in root_path:
        return [root_path]
    
    if recursive:
        for root, _, files in os.walk(root_path):
            for file in files:
                if file.lower().endswith(f'.{format_extension.lower()}'):
                    wsi_files.append(os.path.join(root, file))
    else:
        # Non-recursive mode, just check the top directory
        for file in os.listdir(root_path):
            file_path = os.path.join(root_path, file)
            if os.path.isfile(file_path) and file.lower().endswith(f'.{format_extension.lower()}'):
                wsi_files.append(file_path)
    
    return wsi_files


def NanfangHospital(label_img):
    texts = ocr.readtext(np.array(label_img))
    slide_id = 'failed'
    batch = 'failed'
    for index, item in enumerate(texts):
        if 'PA' in item[1] and len(item[1]) == 9:
            slide_id = item[1]
            batch = texts[index+1][1]
            break
    return slide_id, batch


def Heibeisiyuan(label_img):
    slide_id = 'failed'
    batch = 'failed'
    for angle in [180, 0, 90, 270]:
        img = label_img.copy().rotate(angle, expand=True)
        texts = ocr.readtext(np.array(img))
        stop_flag = False
        for index, item in enumerate(texts):
            if '-' in item[1] and item[2] > 0.6:
                slide_id = item[1]
                batch = texts[index+1][1] if texts[index+1][2] > 0.6 else batch
                stop_flag = True
                break
        if stop_flag:
            break
    return slide_id, batch


def QingyuanHospital(label_img):
    slide_id = 'failed'
    batch = 'failed'
    for angle in [0, 90, 180, 270]:
        img = label_img.copy().rotate(angle, expand=True)
        texts = ocr.readtext(np.array(img))
        stop_flag = False
        for index, item in enumerate(texts):
            if 'B' in item[1] and len(item[1]) == 7:
                slide_id = item[1]
                batch = texts[index+1][1]
                stop_flag = True
                break
        if stop_flag:
            break
    return slide_id, batch


def PWH(label_img):
    slide_id = 'failed'
    batch = 'failed'
    
    # Try different rotations to handle potential orientation issues
    for angle in [0, 90, 180, 270]:
        img = label_img.copy().rotate(angle, expand=True)
        texts = ocr.readtext(np.array(img))        
        # Consolidate detected text for pattern matching
        all_text = " ".join([item[1] for item in texts])
        # Match both regular letters and the number 5 (which might be misread S)
        consolidated_matches = re.findall(r'\d{2}[A-Za-z5]-?\d{4,6}', all_text.replace(" ", ""))
        if consolidated_matches:
            slide_id = consolidated_matches[0].upper()
            # Convert '5' to 'S' only when it's in the letter position
            if re.search(r'\d{2}5-?\d{4,6}', slide_id):
                slide_id = re.sub(r'(\d{2})5(-?\d{4,6})', r'\1S\2', slide_id)
            
            # Look for the batch code
            for batch_item in texts:
                batch_text = batch_item[1].strip()
                # Look for C followed by 3-4 digits
                if re.search(r'C\d{3,4}', batch_text) and batch_item[2] > 0.3:
                    batch = batch_text
                    break
                # Or look for text immediately after "PWH"
                elif "PWH" in batch_text and len(batch_text) > 3:
                    possible_batch = batch_text.replace("PWH", "").strip()
                    if possible_batch:
                        batch = possible_batch
                        break
            break
        
        # Look for text patterns in individual elements if consolidated approach fails
        for index, item in enumerate(texts):
            text = item[1].replace(" ", "")
            # Match both regular letters and the number 5
            match = re.search(r'(\d{2}[A-Za-z5]-?\d{4,6})', text)
            
            if match and item[2] > 0.3:
                slide_id = match.group(1).upper()
                # Convert '5' to 'S' only when it's in the letter position
                if re.search(r'\d{2}5-?\d{4,6}', slide_id):
                    slide_id = re.sub(r'(\d{2})5(-?\d{4,6})', r'\1S\2', slide_id)
                
                # Look for the batch code
                for batch_item in texts:
                    batch_text = batch_item[1].strip()
                    if re.search(r'C\d{3,4}', batch_text) and batch_item[2] > 0.3:
                        batch = batch_text
                        break
                    elif "PWH" in batch_text and len(batch_text) > 3:
                        possible_batch = batch_text.replace("PWH", "").strip()
                        if possible_batch:
                            batch = possible_batch
                            break
                break
        
        # If we found an ID, no need to try other rotations
        if slide_id != 'failed':
            break
    
    return slide_id, batch

def caller(hospital_name, label_img):
    if hospital_name == 'Nanfang':
        return NanfangHospital(label_img)
    elif hospital_name == 'Qingyuan':
        return QingyuanHospital(label_img)
    elif hospital_name == 'Hebeisiyuan':
        return Heibeisiyuan(label_img)
    elif hospital_name == 'PWH':
        return PWH(label_img)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    args = parser()
    print(f"Starting WSI label extraction for {args.hospital} hospital")
    
    # Find all WSI files recursively
    wsi_files = find_wsi_files(args.path, args.format, args.recursive)
    
    print(f"Found {len(wsi_files)} files to process")
    
    # Determine where to save the output CSV
    if args.output:
        # If output is explicitly specified, use that
        output_dir = args.output
        csv_filename = 'id_table.csv'
    else:
        if os.path.isfile(args.path):
            # If path is a file, use its parent directory and the file's name
            output_dir = os.path.dirname(os.path.dirname(args.path))  # parent of parent
            basename = os.path.basename(os.path.dirname(args.path))  # name of parent folder
            csv_filename = f"{basename}_id_table.csv"
        else:
            # If path is a directory, use its parent directory and its name
            output_dir = os.path.dirname(args.path)  # parent directory
            basename = os.path.basename(args.path)   # name of the directory
            csv_filename = f"{basename}_id_table.csv"
    
    # Make sure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    ocr = easyocr.Reader(['en'], gpu=False)
    
    name_dict = {}
    successful = 0
    failed = 0
    
    for index, file_name in enumerate(wsi_files):
        print(f"[{index+1}/{len(wsi_files)}] Processing: {os.path.basename(file_name)}")
        slide_id = os.path.splitext(os.path.basename(file_name))[0]
        
        try:
            handle = openslide.OpenSlide(file_name)
            
            # For mrxs files, the label might be named differently
            label_key = 'label' if 'label' in handle.associated_images else 'macro'
            if label_key not in handle.associated_images and args.hospital == 'PWH':
                available_keys = list(handle.associated_images.keys())
                if available_keys:
                    label_key = available_keys[0]  # Use the first available
                    print(f"Using '{label_key}' as the label image")
                else:
                    print(f"No associated images found for {file_name}")
                    failed += 1
                    continue
            
            img = handle.associated_images[label_key].convert('RGB')
            
            result = caller(args.hospital, img)
            name_dict[slide_id] = result
            
            if result[0] == 'failed':
                print(f"Failed to extract ID for {slide_id}")
                failed += 1
            else:
                print(f"Extracted ID: {result[0]}, Batch: {result[1]}")
                successful += 1
                
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
            failed += 1
    
    # Write the results to the CSV file
    csv_path = os.path.join(output_dir, csv_filename)
    with open(csv_path, 'w') as f:
        f.write('扫描号,病理号,编号\n')
        for sid, (case_id, d) in name_dict.items():
            f.write('{},{},"{}"\n'.format(sid, case_id, d))
    
    print(f"Processing complete! Successful: {successful}, Failed: {failed}")
    print(f"Results saved to: {csv_path}")

# if __name__ == '__main__':
#     args = parser()
#     print(f"Starting WSI label extraction for {args.hospital} hospital")
    
#     # Determine where to save the output CSV
#     if args.output:
#         output_dir = args.output
#     else:
#         # Default to the input path directory if no output specified
#         if os.path.isfile(args.path):
#             output_dir = os.path.dirname(args.path)
#         else:
#             output_dir = args.path
        
#     # Make sure output directory exists
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     ocr = easyocr.Reader(['en'], gpu=False)
    
#     # Find all WSI files recursively
#     wsi_files = find_wsi_files(args.path, args.format, args.recursive)
    
#     print(f"Found {len(wsi_files)} files to process")
    
#     name_dict = {}
#     successful = 0
#     failed = 0
    
#     for index, file_name in enumerate(wsi_files):
#         print(f"[{index+1}/{len(wsi_files)}] Processing: {os.path.basename(file_name)}")
#         slide_id = os.path.splitext(os.path.basename(file_name))[0]
        
#         try:
#             handle = openslide.OpenSlide(file_name)
            
#             # For mrxs files, the label might be named differently
#             label_key = 'label' if 'label' in handle.associated_images else 'macro'
#             if label_key not in handle.associated_images and args.hospital == 'PWH':
#                 available_keys = list(handle.associated_images.keys())
#                 if available_keys:
#                     label_key = available_keys[0]  # Use the first available
#                     print(f"Using '{label_key}' as the label image")
#                 else:
#                     print(f"No associated images found for {file_name}")
#                     failed += 1
#                     continue
            
#             img = handle.associated_images[label_key].convert('RGB')
            
#             result = caller(args.hospital, img)
#             name_dict[slide_id] = result
            
#             if result[0] == 'failed':
#                 print(f"Failed to extract ID for {slide_id}")
#                 failed += 1
#             else:
#                 print(f"Extracted ID: {result[0]}, Batch: {result[1]}")
#                 successful += 1
                
#         except Exception as e:
#             print(f"Error processing {file_name}: {str(e)}")
#             failed += 1
    
#     # Write the results to the CSV file
#     csv_path = os.path.join(output_dir, 'id_table.csv')
#     with open(csv_path, 'w') as f:
#         f.write('扫描号,病理号,编号\n')
#         for sid, (case_id, d) in name_dict.items():
#             f.write('{},{},"{}"\n'.format(sid, case_id, d))
    
#     print(f"Processing complete! Successful: {successful}, Failed: {failed}")
#     print(f"Results saved to: {csv_path}")
