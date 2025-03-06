from transformers import AutoImageProcessor, AutoModel
import torch



def get_model(device, gpu_num):
    processor = AutoImageProcessor.from_pretrained("histai/hibou-L", trust_remote_code=True)
    model = AutoModel.from_pretrained("histai/hibou-L", trust_remote_code=True).to(device)
    model.eval()

    def func(image):
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            features = outputs.last_hidden_state[:, 0, :]
            return features
        
    return func    
    
    
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_num = torch.cuda.device_count()
    print(f"Device: {device}, GPU number: {gpu_num}")
    model = get_model(device, gpu_num)
    print(model)
    print(model(torch.rand((1, 3, 224, 224)).to(device)).shape)