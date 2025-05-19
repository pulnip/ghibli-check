import copy

class Callback:
    def on_train_begin(self, logs=None):
        pass
    def on_epoch_end(self, epoch, logs=None):
        pass
    def on_train_end(self, logs=None):
        pass

class EarlyStopping(Callback):
    def __init__(self, patience=3, restore_best_weights=True):
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.best_val_loss = float('inf')
        self.best_weights = None
        self.wait = 0
    def on_train_begin(self, logs=None):
        self.best_val_loss = float('inf')
        self.wait = 0
        self.best_weights = None
    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')
        model = logs.get('model')
        if val_loss is None or model is None:
            return
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_weights = copy.deepcopy(model.state_dict())
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    print("Model weights restored to best epoch")
                logs['stop_training'] = True
    def on_train_end(self, logs=None):
        pass
