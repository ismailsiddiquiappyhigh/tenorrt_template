with torch.no_grad():
    torch.onnx.export(
        model,  # The dark creation you wish to export
        x,  # A sacrificial input to guide the export
        'llie_dyna_mod.onnx',  # The tome where you'll inscribe your model
        export_params=True,  # Bind the model's learned incantations (parameters)
        opset_version=16,  # The dialect of ONNX you wish to use
        do_constant_folding=True,  # Optimize the model with constant folding
        verbose=True,  # Verbose, because why not? More info is always fun.
        input_names=['input'],  # Name your input, as all things need names
        output_names=['output'],  # And your output too
        dynamic_axes={'input': {0: 'batch_size', 2 : 'height', 3 : 'width'},  # Dynamic batch size, height, and width
                      'output': {0: 'batch_size'}}  # Dynamic batch size for output

    )
