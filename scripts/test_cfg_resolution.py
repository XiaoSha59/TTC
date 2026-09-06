from omegaconf import OmegaConf

cfg = OmegaConf.create({
    'class_ratios': [0.5, 0.5],
    'module': {
        'class_weights': '${class_ratios}'
    }
})

print("Default resolved:", cfg.module.class_weights)

# Override like Hydra CLI
cli = OmegaConf.from_cli(['class_ratios=[0.632,0.368]'])
merged = OmegaConf.merge(cfg, cli)

print("CLI Override resolved:", merged.module.class_weights)
print("Type of element:", type(merged.module.class_weights[0]))
assert float(merged.module.class_weights[0]) == 0.632
assert float(merged.module.class_weights[1]) == 0.368
print("TEST SUCCESS: class_weights perfectly interpolates from class_ratios!")
