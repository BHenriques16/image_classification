import numpy as np
import pandas as pd
import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from models_factory import get_model

# Configs
BASE_DIR = 'dataset_waste_container'  
OUTPUT_CSV = 'relatorio_detalhado.csv'
MODEL_PATH = 'models/resnet18_best.pth'

# Automatically detect classes (alphabetical order, PyTorch default)
# This is crucial so that index 0 corresponds to the same training class.
CLASS_NAMES = sorted([d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))])
NUM_CLASSES = len(CLASS_NAMES)

print(f"Classes detetadas ({NUM_CLASSES}): {CLASS_NAMES}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def load_model():
    print(f"Carregando modelo ResNet18 de {MODEL_PATH}...")
    model = get_model("resnet18", NUM_CLASSES)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print(f"ERRO CRÍTICO: Modelo não encontrado em {MODEL_PATH}")
    model.to(device)
    model.eval()
    return model

model = load_model()
transform = get_transforms()

def classifier_predict(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = F.softmax(outputs, dim=1)
            
        max_prob, pred_idx = torch.max(probs, dim=1)
        return pred_idx.item(), max_prob.item()
        
    except Exception as e:
        print(f"Erro na imagem {image_path}: {e}")
        return 0, 0.0

def run_test():
    results_list = []
    
    print(f"A iniciar avaliação em '{BASE_DIR}'...")

    # Go through each folder to find the "Real_Folder" (Ground Truth).
    for class_folder in CLASS_NAMES:
        folder_path = os.path.join(BASE_DIR, class_folder)
        if not os.path.isdir(folder_path): continue

        files = os.listdir(folder_path)
        
        for filename in files:
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue

            filepath = os.path.join(folder_path, filename)
            
            pred_idx, max_prob = classifier_predict(filepath)
            
            pred_class_name = CLASS_NAMES[pred_idx]
            
            is_correct = (class_folder == pred_class_name)
            acerto_str = f"• {is_correct}"
            
            row = {
                'Imagem': filename,
                'Pasta_Real': class_folder,      
                'Previsão_Classe': pred_class_name, 
                'Probabilidade_Max': f"{max_prob:.7f}", 
                'Acerto': acerto_str
            }
            results_list.append(row)

    df_results = pd.DataFrame(results_list)
    
    df_results.to_csv(OUTPUT_CSV, index=False)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'left')

    print(f"\nProcessamento concluído. Analisadas {len(df_results)} imagens.")
    print(f"Resultados guardados em: {OUTPUT_CSV}")
    print("\n--- Amostra dos Resultados ---")
    print(df_results.head(10))

    if len(df_results) > 0:
        total_acc = df_results['Acerto'].str.contains('True').mean() * 100
        print(f"\nAccuracy Global neste conjunto: {total_acc:.2f}%")

if __name__ == '__main__':
    run_test()