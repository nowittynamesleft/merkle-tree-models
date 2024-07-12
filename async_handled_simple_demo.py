import asyncio
import json
import os
import subprocess
import sys
import torch
from torch import nn
import ezkl

# Install dependencies if running in Google Colab
try:
    import google.colab
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ezkl"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "onnx"])
except:
    pass

# Define the model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=3, kernel_size=5, stride=2)
        self.relu = nn.ReLU()
        self.d1 = nn.Linear(48, 48)
        self.d2 = nn.Linear(48, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.flatten(start_dim=1)
        x = self.d1(x)
        x = self.relu(x)
        logits = self.d2(x)
        return logits

def slerp(val, low, high):
    omega = torch.acos(torch.clamp(torch.sum(low/torch.norm(low)*high/torch.norm(high)), -1, 1))
    so = torch.sin(omega)
    return torch.sin((1.0-val)*omega) / so * low + torch.sin(val*omega) / so * high

class MetaModel(nn.Module):
    def __init__(self, model1, model2, merge_type='average'):
        super(MetaModel, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model = MyModel()
        self.merge_type = merge_type
        self.m1_weight = 0.5
        self.m2_weight = 0.5
        self._merge_parameters()

    def _merge_parameters(self):
        for param1, param2, param in zip(self.model1.parameters(), self.model2.parameters(), self.model.parameters()):
            if self.merge_type == 'slerp':
                param.data.copy_(slerp(self.m1_weight, param1.data, param2.data))
            else:
                param.data.copy_(self.m1_weight * param1.data + self.m2_weight * param2.data)

    def forward(self, x):
        return self.model(x)

async def main():
    # Instantiate two models
    model1 = MyModel()
    model2 = MyModel()

    # Load pre-trained weights if available
    # model1.load_state_dict(torch.load('path_to_model1_weights.pth'))
    # model2.load_state_dict(torch.load('path_to_model2_weights.pth'))

    # Instantiate the MetaModel with 'average' or 'slerp' merging
    circuit = MetaModel(model1, model2, merge_type='slerp')

    model_path = os.path.join('network.onnx')
    compiled_model_path = os.path.join('network.compiled')
    pk_path = os.path.join('test.pk')
    vk_path = os.path.join('test.vk')
    settings_path = os.path.join('settings.json')
    witness_path = os.path.join('witness.json')
    data_path = os.path.join('input.json')
    shape = [1, 28, 28]

    # After training, export to onnx (network.onnx) and create a data file (input.json)
    x = 0.1 * torch.rand(1, *shape, requires_grad=True)
    circuit.eval()

    # Export the model
    torch.onnx.export(circuit, x, model_path, export_params=True, opset_version=10, do_constant_folding=True,
                      input_names=['input'], output_names=['output'], dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

    data_array = x.detach().numpy().reshape([-1]).tolist()
    data = dict(input_data=[data_array])
    json.dump(data, open(data_path, 'w'))

    py_run_args = ezkl.PyRunArgs()
    py_run_args.input_visibility = "public"
    py_run_args.output_visibility = "public"
    py_run_args.param_visibility = "fixed"

    res = ezkl.gen_settings(model_path, settings_path, py_run_args=py_run_args)
    assert res == True

    cal_path = os.path.join("calibration.json")
    data_array = torch.rand(20, *shape, requires_grad=True).detach().numpy().reshape([-1]).tolist()
    data = dict(input_data=[data_array])
    json.dump(data, open(cal_path, 'w'))

    await ezkl.calibrate_settings(cal_path, model_path, settings_path, "resources")

    res = ezkl.compile_circuit(model_path, compiled_model_path, settings_path)
    assert res == True

    # Ensure the SRS is available
    srs_path = os.path.expanduser("~/.ezkl/srs/kzg14.srs")
    if not os.path.isfile(srs_path):
        print("Downloading SRS...")
        res = await ezkl.get_srs(settings_path)
        print(f"SRS download result: {res}")
        if res != True:
            raise RuntimeError(f"Failed to download SRS: {res}")
        assert res == True

    res = await ezkl.gen_witness(data_path, compiled_model_path, witness_path)
    assert os.path.isfile(witness_path)

    res = ezkl.setup(compiled_model_path, vk_path, pk_path)
    if res != True:
        raise RuntimeError(f"Setup failed: {res}")
    assert res == True
    assert os.path.isfile(vk_path)
    assert os.path.isfile(pk_path)
    assert os.path.isfile(settings_path)

    proof_path = os.path.join('test.pf')
    res = ezkl.prove(witness_path, compiled_model_path, pk_path, proof_path, "single")
    print(res)
    if res != True:
        raise RuntimeError("Prove failed")
    assert os.path.isfile(proof_path)

    res = ezkl.verify(proof_path, settings_path, vk_path)
    if res != True:
        raise RuntimeError("Verify failed")
    assert res == True
    print("verified")

# Run the main function in an event loop
if __name__ == "__main__":
    asyncio.run(main())

