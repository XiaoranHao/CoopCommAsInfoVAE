import torch
import os


def train(model, train_loader, args, device, log_interval=1):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-4)
    for epoch in range(1, args.epochs + 1):
        loss_sum = 0
        for batch_idx, (data, target) in enumerate(train_loader):
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

