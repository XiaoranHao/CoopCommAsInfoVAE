import torch
import os


def CreatePath(DicName):
    cwd = os.getcwd()
    return os.path.join(cwd, DicName)


def SaveModel(model, DicName, filename):
    path = CreatePath(DicName)
    if not os.path.isdir(path):
        os.mkdir(path)
        print("dictionary does not exit, create a new one")
    if not (filename.endswith('.pth') or filename.endswith('.pt')):
        raise RuntimeError("Wrong extension for pytorch saving")
    full_name = os.path.join(path, filename)
    torch.save(model.state_dict(), full_name)


def train(model, train_loader, args, device, log_interval=1):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs/10, eta_min=1e-4)
    for epoch in range(1, args.epochs + 1):
        loss_sum = 0
        for batch_idx, (data, target, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            P, C, kl, z = model(data)
            loss = model.EotLoss(P, C, kl)
            loss.backward()
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                loss_sum += loss.item()

            if epoch % log_interval == 0 and batch_idx == len(train_loader)-1:
                # print(f"Epoch: {epoch}, Loss: {loss[0].item()}, Reconstruction: {loss[1].item()}, "
                #       f"kl_z2: {loss[2].item()}")
                print(f"Epoch: {epoch}, Loss: {loss_sum / len(train_loader)}")
    if not (os.path.isdir('./savedmodels')):
        os.mkdir('./savedmodels')
        print("not exist")
    torch.save(model.state_dict(), './savedmodels/' + args.save + '.pth')
    print("after save")


def train_DualOT(model, train_loader, args, device, log_interval=1):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    for epoch in range(1, args.epochs + 1):
        loss_sum = 0
        for batch_idx, (data, target, idx) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            C_, C, z_, z = model(data, idx)
            loss = model.EotLoss(C)
            loss.backward()
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                loss_sum += loss.item()

            if epoch % log_interval == 0 and batch_idx == len(train_loader)-1:
                # print(f"Epoch: {epoch}, Loss: {loss[0].item()}, Reconstruction: {loss[1].item()}, "
                #       f"kl_z2: {loss[2].item()}")
                print(f"Epoch: {epoch}, Loss: {loss_sum / len(train_loader)}")
    SaveModel(model, "savedmodels", args.save)


def train_DualOT2(model, train_loader, data_all, args, device, log_interval=1):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    for epoch in range(1, args.epochs + 1):
        loss_sum = 0
        z_all = model.z_sample(args.n_samples).to(device)
        C_all = model.make_cost(data_all.to(device), z_all).to('cpu')
        for i in range(1, 1 + args.n_iter):
            x_index = torch.arange(0,2000)
            z_index = torch.randint(0, args.n_samples, (args.z_bs,))
            z = z_all[z_index]
            C = C_all[x_index][:,z_index].to(device)
            model.DualOT.learn_OT(x_index, z, C)
            if i % 1000 == 0:
                print(f"Iters {i}/{args.n_iter}")
                model.DualOT(x_index, z, C)
        print("OT Done")
        for batch_idx, (data, target, idx) in enumerate(train_loader):
            data = data.to(device)
            C_batch = C_all[idx].to(device)
            optimizer.zero_grad()
            C_, z_ = model(data, idx, z_all, C_batch)
            loss = model.DecLoss(C_)
            loss.backward()
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                loss_sum += loss.item()

            if epoch % log_interval == 0 and batch_idx == len(train_loader)-1:
                # print(f"Epoch: {epoch}, Loss: {loss[0].item()}, Reconstruction: {loss[1].item()}, "
                #       f"kl_z2: {loss[2].item()}")
                print(f"Epoch: {epoch}, Loss: {loss_sum / len(train_loader)}")
        SaveModel(model, "savedmodels", args.save)

def train_SemiDualOT(model, train_loader, args, device, log_interval=1):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    for epoch in range(1, args.epochs + 1):
        loss_sum = 0
        for batch_idx, (data, target, idx) in enumerate(train_loader):
            optimizer.zero_grad()
            C, z = model(idx)
            loss = model.EotLoss(C)
            loss.backward()
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                loss_sum += loss.item()

            if epoch % log_interval == 0 and batch_idx == len(train_loader)-1:
                # print(f"Epoch: {epoch}, Loss: {loss[0].item()}, Reconstruction: {loss[1].item()}, "
                #       f"kl_z2: {loss[2].item()}")
                print(f"Epoch: {epoch}, Loss: {loss_sum / len(train_loader)}")
        SaveModel(model, "savedmodels", args.save)


def train_VAE(model, train_loader, args, device, log_interval=1):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    for epoch in range(1, args.epochs + 1):
        loss_sum = 0
        recon_sum = 0
        kl_sum = 0
        for batch_idx, (data, target, idx) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss, recon, kl = model.loss(data, *output)
            loss.backward()
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                loss_sum += loss.item()
                recon_sum += recon.item()
                kl_sum += kl.item()

            if epoch % log_interval == 0 and batch_idx == len(train_loader)-1:
                print(f"Epoch: {epoch}, Loss: {loss_sum / len(train_loader)}, Reconstruction:"
                      f" {recon_sum / len(train_loader)}, "
                      f"kl_z: {kl_sum / len(train_loader)}")
    SaveModel(model.decoder, "savedmodels", args.save)
    SaveModel(model, "savedmodels", "vae"+ str(args.seed)+".pth")

