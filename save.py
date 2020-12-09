import os
from datetime import datetime

if not os.path.exists('./models/'):
    os.mkdir('./models/')
    
if not os.path.exists('./models/checkpoints/'):
    os.mkdir('./models/checkpoints/')
    
if not os.path.exists('./models/finished/'):
    os.mkdir('./models/finished/')
    
def save_model(model, batch_size, data_pct, epoch=None):
    path = './models/'
    if epoch is not None:
        path += 'checkpoints/'
    else:
        path += 'finished/'
        
    file_name = f'{path}bs-{batch_size}_pct-{data_pct}_e-{"None" if epoch is None else epoch}_{datetime.now()}'
    torch.save(model.state_dict(), file_name)  