import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from datasets.cell.tabula_muris import *
from utils.io_utils import get_resume_file, hydra_setup, fix_seed


def initialize_dataset_model(cfg):
    # Instantiate train dataset as specified in dataset config under simple_cls or set_cls
    if cfg.method.type == "baseline":
        train_dataset = instantiate(cfg.dataset.simple_cls, batch_size=cfg.method.train_batch, mode='train')
    elif cfg.method.type == "meta":
        train_dataset = instantiate(cfg.dataset.set_cls, mode='train')
    else:
        raise ValueError(f"Unknown method type: {cfg.method.type}")
    train_loader = train_dataset.get_data_loader()

    # Instantiate val dataset as specified in dataset config under simple_cls or set_cls
    # Eval type (simple or set) is specified in method config, rather than dataset config
    if cfg.method.eval_type == 'simple':
        val_dataset = instantiate(cfg.dataset.simple_cls, batch_size=cfg.method.val_batch, mode='val')
    else:
        val_dataset = instantiate(cfg.dataset.set_cls, mode='val')
    val_loader = val_dataset.get_data_loader()

    # For MAML (and other optimization-based methods), need to instantiate backbone layers with fast weight
    if cfg.method.fast_weight:
        backbone = instantiate(cfg.backbone, x_dim=train_dataset.dim, fast_weight=True)
    else:
        backbone = instantiate(cfg.backbone, x_dim=train_dataset.dim)

    # Instantiate few-shot method class
    model = instantiate(cfg.method.cls, backbone=backbone)

    if torch.cuda.is_available():
        model = model.cuda()

    if cfg.method.name == 'maml':
        cfg.method.stop_epoch *= model.n_task  # maml use multiple tasks in one update

    return train_loader, val_loader, model


@hydra.main(version_base=None, config_path='conf', config_name='main')
def run(cfg):
    print(OmegaConf.to_yaml(cfg, resolve=True))

    model = None

    fix_seed(cfg.exp.seed)
    
    modes = ["train", "val", "test"]
    
    for mode in modes:
                
        dataset = instantiate(cfg.dataset.set_cls, mode=mode)
        loader = dataset.get_data_loader()
        
        if model is None:
            # For MAML (and other optimization-based methods), need to instantiate backbone layers with fast weight
            if cfg.method.fast_weight:
                backbone = instantiate(cfg.backbone, x_dim=dataset.dim, fast_weight=True)
            else:
                backbone = instantiate(cfg.backbone, x_dim=dataset.dim)

            # Instantiate few-shot method class
            model = instantiate(cfg.method.cls, backbone=backbone)

            if torch.cuda.is_available():
                model = model.cuda()

            if cfg.method.name == 'maml':
                cfg.method.stop_epoch *= model.n_task 
        
            resume_file = get_resume_file("cfg.path")
            if resume_file is not None:
                tmp = torch.load(resume_file)
                model.load_state_dict(tmp['state'])

            model.eval()

        features = []
        for (x, _) in loader:
            x = x.cuda().contiguous().view(-1, *x.size()[2:])
            print(mode, x.shape)
            with torch.no_grad():
                features.append(model.forward(x).cpu())
        
        features = torch.cat(features, dim=0)            
        torch.save(features, f"{cfg.exp.name}_{mode}_features.pt")
        print(f"Saved features to: {cfg.exp.name}_{mode}_features.pt")


if __name__ == '__main__':
    hydra_setup()
    run()
