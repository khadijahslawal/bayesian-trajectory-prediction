from src.data_loader import ScenesDataLoader
loader = ScenesDataLoader(data_root='data/raw/')

train_loader = loader.get_train_loader(
    scenes=['eth', 'hotel', 'univ', 'zara1', 'students1']
)

# val_loader = loader.get_val_loader(scene='zara2')
# test_loader = loader.get_test_loader(scene='zara2')

print(f"Training samples: {len(train_loader.dataset)}")
# print(f"Test samples: {len(test_loader.dataset)}")
