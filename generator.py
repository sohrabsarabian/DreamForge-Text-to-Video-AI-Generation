import torch
import math
from models.clip_model import CLIPModel
from models.taming_model import TamingModel
from parameters import Parameters
from utils.image_processing import norm_data, create_crops, show_from_tensor
from utils.video_creation import create_video


class ImageGenerator:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.clip_model = CLIPModel(self.device)
        self.taming_model = TamingModel(args.taming_config_path, args.taming_checkpoint_path, self.device)
        self.params, self.optimizer = self.init_params()
        self.aug_transform = torch.nn.Sequential(
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomAffine(30, (.2, .2), fill=0)
        ).to(self.device)

    def init_params(self):
        params = Parameters(self.args.batch_size, self.args.size1, self.args.size2).to(self.device)
        optimizer = torch.optim.AdamW([{'params': [params.data], 'lr': self.args.learning_rate}],
                                      weight_decay=self.args.wd)
        return params, optimizer

    def generate(self):
        res_img = []
        res_z = []

        for prompt in self.args.include:
            iteration = 0
            self.params, self.optimizer = self.init_params()

            for it in range(self.args.total_iter):
                loss = self.optimize(prompt)

                if iteration >= 80 and iteration % self.args.show_step == 0:
                    new_img = self.showme(self.params, show_crop=False)
                    res_img.append(new_img)
                    res_z.append(self.params())
                    print(f"loss: {loss.item()}, iteration: {iteration}")

                iteration += 1
            torch.cuda.empty_cache()
        return res_img, res_z

    def optimize(self, prompt):
        out = self.taming_model.generate(self.params())
        out = norm_data(out)
        out = create_crops(out, num_crops=32, size1=self.args.size1, aug_transform=self.aug_transform,
                           noise_factor=self.args.noise_factor)
        out = self.clip_model.normalize(out)
        image_enc = self.clip_model.encode_image(out)

        text_enc = self.clip_model.encode_text(prompt)
        extras_enc = self.clip_model.encode_text(self.args.extras) if self.args.extras else 0
        exclude_enc = self.clip_model.encode_text(self.args.exclude) if self.args.exclude else 0

        final_enc = self.args.w1 * text_enc + self.args.w1 * extras_enc
        final_text_include_enc = final_enc / final_enc.norm(dim=-1, keepdim=True)
        final_text_exclude_enc = exclude_enc

        main_loss = torch.cosine_similarity(final_text_include_enc, image_enc, -1)
        penalize_loss = torch.cosine_similarity(final_text_exclude_enc, image_enc, -1)

        final_loss = -self.args.alpha * main_loss + self.args.beta * penalize_loss

        loss = final_loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def interpolate(self, res_z_list, duration_list):
        gen_img_list = []
        fps = 25

        for idx, (z, duration) in enumerate(zip(res_z_list, duration_list)):
            num_steps = int(duration * fps)
            z1 = z
            z2 = res_z_list[(idx + 1) % len(res_z_list)]

            for step in range(num_steps):
                alpha = math.sin(1.5 * step / num_steps) ** 6
                z_new = alpha * z2 + (1 - alpha) * z1

                new_gen = norm_data(self.taming_model.generate(z_new).cpu())[0]
                new_img = torchvision.transforms.ToPILImage(mode='RGB')(new_gen)
                gen_img_list.append(new_img)

        return gen_img_list

    def showme(self, params, show_crop):
        with torch.no_grad():
            generated = self.taming_model.generate(params())

            if show_crop:
                print("Augmented cropped example")
                aug_gen = generated.float()
                aug_gen = create_crops(aug_gen, num_crops=1, size1=self.args.size1, aug_transform=self.aug_transform,
                                       noise_factor=self.args.noise_factor)
                aug_gen_norm = norm_data(aug_gen[0])
                show_from_tensor(aug_gen_norm)

            print("Generation")
            latest_gen = norm_data(generated.cpu())
            show_from_tensor(latest_gen[0])

        return latest_gen[0]

    def create_video(self, interp_result_img_list, output_path):
        create_video(interp_result_img_list, output_path)
