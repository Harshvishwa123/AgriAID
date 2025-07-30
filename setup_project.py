#!/usr/bin/env python3
"""
AgriAid Project Setup Script
This script helps you set up the complete AgriAid project structure.
"""

import os
import shutil
import sys

def create_directory_structure():
    """Create the required directory structure"""
    directories = [
        'templates',
        'static/css',
        'static/js',
        'static/uploads',
        'models',
        'datasets'
    ]
    
    print("🏗️  Creating directory structure...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   ✅ Created: {directory}")

def check_models():
    """Check if required model files exist"""
    print("\n🔍 Checking for trained models...")
    
    required_files = [
        'models/plant_disease_model.h5',
        'models/class_labels.json'
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   ✅ Found: {file_path}")
        else:
            print(f"   ❌ Missing: {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  Missing {len(missing_files)} required model files!")
        print("   Please ensure you have trained the disease detection model first.")
        print("   Run your training script (paste.txt) to generate these files.")
        return False
    return True

def check_dataset():
    """Check if dataset files exist"""
    print("\n📊 Checking for datasets...")
    
    # Check PlantVillage dataset
    plant_village_path = r"C:\Users\hvish\Desktop\AGRIAID\PlantVillage"
    if os.path.exists(plant_village_path):
        class_dirs = [d for d in os.listdir(plant_village_path) 
                     if os.path.isdir(os.path.join(plant_village_path, d))]
        print(f"   ✅ PlantVillage dataset found with {len(class_dirs)} classes")
    else:
        print(f"   ❌ PlantVillage dataset not found at: {plant_village_path}")
        print("   Please ensure the dataset path is correct in app.py")
    
    # Check crop recommendation dataset
    crop_csv_path = "datasets/crop_recommendation.csv"
    if os.path.exists(crop_csv_path):
        print(f"   ✅ Crop recommendation dataset found")
    else:
        print(f"   ❌ Missing: {crop_csv_path}")
        print("   Please place your crop_recommendation.csv file in the datasets folder")
        return False
    
    return True

def copy_models_if_exist():
    """Copy model files from current directory if they exist"""
    print("\n📁 Looking for existing model files...")
    
    model_files = [
        ('plant_disease_model.h5', 'models/plant_disease_model.h5'),
        ('class_labels.json', 'models/class_labels.json')
    ]
    
    for source, destination in model_files:
        if os.path.exists(source):
            shutil.copy2(source, destination)
            print(f"   ✅ Copied: {source} → {destination}")
        else:
            print(f"   ❓ Not found in current directory: {source}")

def create_sample_csv():
    """Create a sample crop recommendation CSV if it doesn't exist"""
    csv_path = "datasets/crop_recommendation.csv"
    if not os.path.exists(csv_path):
        print(f"\n📝 Creating sample {csv_path}...")
        
        # Create sample data
        sample_data = """N,P,K,temperature,humidity,ph,rainfall,label
90,42,43,20.879744,82.002744,6.502985,202.935536,rice
85,58,41,21.770462,80.319644,7.038096,226.655537,rice
60,55,44,23.004459,82.320763,7.840207,263.964248,rice
74,35,40,26.491096,80.158363,6.980401,242.864034,rice
78,42,42,20.130175,81.604873,7.628473,262.717340,rice
69,37,42,23.042149,83.034296,7.073017,251.014377,rice
69,55,44,22.726346,82.869765,7.215608,271.364071,rice
94,53,48,21.553465,89.516834,7.283503,278.965570,maize
78,65,48,24.906127,90.395105,7.263707,262.168466,maize
90,62,40,21.977210,90.445127,6.854779,221.194321,maize"""
        
        with open(csv_path, 'w') as f:
            f.write(sample_data)
        print(f"   ✅ Created sample dataset with 10 rows")
        print("   ⚠️  Replace this with your actual crop_recommendation.csv")

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    
    if not os.path.exists('requirements.txt'):
        print("   ❌ requirements.txt not found!")
        return False
    
    try:
        import subprocess
        result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✅ All packages installed successfully!")
        else:
            print("   ❌ Error installing packages:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    return True

def main():
    """Main setup function"""
    print("🌱 AgriAid Project Setup")
    print("=" * 50)
    
    # Create directory structure
    create_directory_structure()
    
    # Copy existing models if available
    copy_models_if_exist()
    
    # Check for required files
    models_ok = check_models()
    dataset_ok = check_dataset()
    
    # Create sample CSV if needed
    if not dataset_ok:
        create_sample_csv()
    
    # Install requirements
    print("\n🤔 Do you want to install required packages? (y/n): ", end="")
    if input().lower().startswith('y'):
        install_ok = install_requirements()
    else:
        install_ok = True
        print("   ⏭️  Skipped package installation")
    
    # Final summary
    print("\n" + "=" * 50)
    print("🎯 SETUP SUMMARY")
    print("=" * 50)
    
    print(f"📁 Directory structure: ✅")
    print(f"🤖 Disease model: {'✅' if models_ok else '❌'}")
    print(f"📊 Datasets: {'✅' if dataset_ok else '⚠️ '}")
    print(f"📦 Packages: {'✅' if install_ok else '❌'}")
    
    if models_ok and dataset_ok and install_ok:
        print("\n🎉 Setup completed successfully!")
        print("\n🚀 To start the application:")
        print("   python app.py")
        print("\n🌐 Then open: http://localhost:5000")
    else:
        print("\n⚠️  Setup completed with issues:")
        if not models_ok:
            print("   • Train the disease detection model first")
        if not dataset_ok:
            print("   • Add your crop_recommendation.csv to datasets folder")
        if not install_ok:
            print("   • Install required packages manually: pip install -r requirements.txt")
    
    print("\n📋 Next steps:")
    print("   1. Ensure your PlantVillage dataset path is correct in app.py")
    print("   2. Replace sample data with your actual crop_recommendation.csv")
    print("   3. Train the disease detection model if not already done")
    print("   4. Run the application: python app.py")

if __name__ == "__main__":
    main()