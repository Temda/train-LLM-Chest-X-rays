import os
import pandas as pd
from PIL import Image
import sys
from tqdm import tqdm

def check_image_integrity(csv_file, img_dir):
    print(f"Loading CSV file: {csv_file}")
    
    try:
        df = pd.read_csv(csv_file)
        print(f"CSV loaded successfully with {len(df)} entries")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None
    
    if len(df.columns) == 0:
        print("CSV file is empty")
        return None
    
    image_column = df.columns[0]
    print(f"Checking images from column: '{image_column}'")
    
    if not os.path.exists(img_dir):
        print(f"Image directory not found: {img_dir}")
        return None
    
    missing_files = []
    corrupted_files = []
    valid_files = []
    invalid_extensions = []
    
    print(f"\n🔍 Checking {len(df)} images...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Checking images"):
        image_name = row[image_column]
        image_path = os.path.join(img_dir, image_name)
        
        if not os.path.exists(image_path):
            missing_files.append({
                'index': idx,
                'filename': image_name,
                'path': image_path
            })
            continue
        
        _, ext = os.path.splitext(image_name.lower())
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        if ext not in valid_extensions:
            invalid_extensions.append({
                'index': idx,
                'filename': image_name,
                'extension': ext
            })
            continue
        
        try:
            with Image.open(image_path) as img:
                img.verify()
            
            with Image.open(image_path) as img:
                width, height = img.size
                channels = len(img.getbands()) if hasattr(img, 'getbands') else 'Unknown'
                
                valid_files.append({
                    'index': idx,
                    'filename': image_name,
                    'width': width,
                    'height': height,
                    'channels': channels,
                    'mode': img.mode,
                    'size_mb': os.path.getsize(image_path) / (1024 * 1024)
                })
                
        except Exception as e:
            corrupted_files.append({
                'index': idx,
                'filename': image_name,
                'error': str(e)
            })
    
    results = {
        'total_entries': len(df),
        'valid_images': len(valid_files),
        'missing_files': len(missing_files),
        'corrupted_files': len(corrupted_files),
        'invalid_extensions': len(invalid_extensions),
        'valid_files_list': valid_files,
        'missing_files_list': missing_files,
        'corrupted_files_list': corrupted_files,
        'invalid_extensions_list': invalid_extensions
    }
    
    return results

def print_summary(results):

    if results is None:
        return
    
    print("\n" + "="*60)
    print("📊 IMAGE VALIDATION SUMMARY")
    print("="*60)
    
    print(f"Total entries in CSV: {results['total_entries']}")
    print(f"Valid images: {results['valid_images']}")
    print(f"Missing files: {results['missing_files']}")
    print(f"Corrupted files: {results['corrupted_files']}")
    print(f"Invalid extensions: {results['invalid_extensions']}")
    
    if results['total_entries'] > 0:
        valid_percent = (results['valid_images'] / results['total_entries']) * 100
        print(f"Success rate: {valid_percent:.2f}%")
    
    if results['missing_files'] > 0:
        print(f"\n Missing files ({results['missing_files']}):")
        for item in results['missing_files_list'][:5]:  
            print(f"   - Index {item['index']}: {item['filename']}")
        if results['missing_files'] > 5:
            print(f"   ... and {results['missing_files'] - 5} more missing files")
    
    if results['corrupted_files'] > 0:
        print(f"\n🔧 Corrupted files ({results['corrupted_files']}):")
        for item in results['corrupted_files_list'][:5]:
            print(f"   - Index {item['index']}: {item['filename']} - {item['error']}")
        if results['corrupted_files'] > 5:
            print(f"   ... and {results['corrupted_files'] - 5} more corrupted files")
    
    if results['invalid_extensions'] > 0:
        print(f"\n Invalid extensions ({results['invalid_extensions']}):")
        for item in results['invalid_extensions_list'][:5]:
            print(f"   - Index {item['index']}: {item['filename']} ({item['extension']})")
        if results['invalid_extensions'] > 5:
            print(f"   ... and {results['invalid_extensions'] - 5} more invalid extensions")
    
    if results['valid_images'] > 0:
        valid_files = results['valid_files_list']
        
        print(f"\n Valid images statistics:")
        
        widths = [img['width'] for img in valid_files]
        heights = [img['height'] for img in valid_files]
        sizes_mb = [img['size_mb'] for img in valid_files]
        
        print(f"   Image dimensions:")
        print(f"      - Width: min={min(widths)}, max={max(widths)}, avg={sum(widths)/len(widths):.1f}")
        print(f"      - Height: min={min(heights)}, max={max(heights)}, avg={sum(heights)/len(heights):.1f}")
        print(f"   File sizes: min={min(sizes_mb):.3f}MB, max={max(sizes_mb):.3f}MB, avg={sum(sizes_mb)/len(sizes_mb):.3f}MB")
        
        modes = {}
        for img in valid_files:
            mode = img['mode']
            modes[mode] = modes.get(mode, 0) + 1
        
        print(f" Image modes:")
        for mode, count in modes.items():
            print(f"      - {mode}: {count} images ({count/len(valid_files)*100:.1f}%)")

def save_report(results, output_file):
    if results is None:
        return
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("IMAGE VALIDATION REPORT\\n")
        f.write("="*50 + "\\n\\n")
        
        f.write(f"Total entries: {results['total_entries']}\\n")
        f.write(f"Valid images: {results['valid_images']}\\n")
        f.write(f"Missing files: {results['missing_files']}\\n")
        f.write(f"Corrupted files: {results['corrupted_files']}\\n")
        f.write(f"Invalid extensions: {results['invalid_extensions']}\\n\\n")
        
        if results['missing_files'] > 0:
            f.write("MISSING FILES:\\n")
            for item in results['missing_files_list']:
                f.write(f"Index {item['index']}: {item['filename']}\\n")
            f.write("\\n")
        
        if results['corrupted_files'] > 0:
            f.write("CORRUPTED FILES:\\n")
            for item in results['corrupted_files_list']:
                f.write(f"Index {item['index']}: {item['filename']} - {item['error']}\\n")
            f.write("\\n")
        
        if results['invalid_extensions'] > 0:
            f.write("INVALID EXTENSIONS:\\n")
            for item in results['invalid_extensions_list']:
                f.write(f"Index {item['index']}: {item['filename']} ({item['extension']})\\n")
            f.write("\\n")
    
    print(f" Report saved to: {output_file}")

def create_clean_csv(results, original_csv, output_csv):
    if results is None or results['valid_images'] == 0:
        print("No valid images to create clean CSV")
        return
    
    df_original = pd.read_csv(original_csv)
    
    valid_indices = [item['index'] for item in results['valid_files_list']]
    df_clean = df_original.iloc[valid_indices].reset_index(drop=True)
    
    df_clean.to_csv(output_csv, index=False)
    print(f"Clean CSV saved to: {output_csv}")
    print(f"   Original: {len(df_original)} entries")
    print(f"   Clean: {len(df_clean)} entries")
    print(f"   Removed: {len(df_original) - len(df_clean)} entries")

def main():
    CSV_PATH = r"D:\\train-LLM-Chest-X-rays\\archive\\new_labels.csv"
    IMG_DIR = r"D:\\train-LLM-Chest-X-rays\\archive\\resized_images\\resized_images"
    
    print("Chest X-ray Dataset Image Validator")
    print("="*50)
    
    results = check_image_integrity(CSV_PATH, IMG_DIR)
    
    print_summary(results)
    
    if results is not None:
        save_report(results, "image_validation_report.txt")
        

        total_invalid = results['missing_files'] + results['corrupted_files'] + results['invalid_extensions']
        if total_invalid > 0:
            print(f"\\n Found {total_invalid} invalid entries. Creating clean CSV...")
            create_clean_csv(results, CSV_PATH, "archive/clean_labels.csv")
        else:
            print("\\n All images are valid! No need to create clean CSV.")

if __name__ == "__main__":
    main()