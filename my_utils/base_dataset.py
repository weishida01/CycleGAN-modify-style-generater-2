import numpy as np
from my_utils import get_calib
import torch
from PIL import Image


# 加载kitti 插入人后的obj点云 返回points, faces
def load_obj_points(obj_path):
    with open(obj_path) as file:
        points = []
        faces = []
        while 1:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "v":
                points.append((float(strs[1]), float(strs[2]), float(strs[3]), float(strs[4])))
            if strs[0] == "f":
                faces.append((int(strs[1]), int(strs[2]), int(strs[3]), float(strs[4])))
    points = np.array(points)

    return points


def obj_img(obj_path,dataroot):
    in_points = load_obj_points(obj_path)
    calib_txt = dataroot + '/calib/' + obj_path.split('/')[-1].split('.')[0].split('_')[0] + '.txt'

    image_size = (384, 512)  # (1280,384)

    # 8、雷达坐标 转 图像坐标
    calib = get_calib.Calibration(calib_txt)
    in_points_img_3, in_points_img_depth = calib.lidar_to_img(in_points[:,:3])
    in_points_img_4 = np.hstack((                                               # x，y,，深度，反射强度
                                  in_points_img_3[:,0].reshape(-1,1),
                                  in_points_img_3[:,1].reshape(-1,1),
                                  in_points_img_depth.reshape(-1,1),
                                  in_points[:,3].reshape(-1,1)
                                  ))

    # 9、图像坐标，转换为图像格式   img_png.size+2 *4  4=x,y,深度，反射强度
    in_img_temp = np.zeros((image_size[1],image_size[0],1), dtype=np.uint8)    # 图像模板
    mask = np.zeros((image_size[1],image_size[0]), dtype=np.uint8)
    # in_img_temp = np.zeros((image_size[1],image_size[0],4), dtype=np.float32) + 10   # 图像模板


    box_center = (in_points_img_4[:,0].mean(),in_points_img_4[:,1].mean())

    for index_point in range(len(in_points_img_4)):
        point = in_points_img_4[index_point]
        x_index = int(point[0] + image_size[0]/2 - box_center[0] + 0.5)
        y_index = int(point[1] + image_size[1]/2 - box_center[1] + 0.5)
        try:
            # in_img_temp[y_index,x_index] = point
            in_img_temp[y_index,x_index] = 255
            kernal = 5
            zero = kernal//2
            for i in range(kernal):
                for j in range(kernal):
                    mask[y_index-zero+i, x_index-zero+j] = 255
        except:
            pass
            # print([y_index,x_index])



    # # 10、图像数值范围变换
    # # /1280 /384 /100 /1
    # in_img_temp[:,:,0] = in_img_temp[:,:,0]/2000
    # in_img_temp[:,:,1] = in_img_temp[:,:,1]/1000
    # in_img_temp[:,:,2] = in_img_temp[:,:,2]/150

    # in_img_temp[in_img_temp > 1] = 1

    return in_img_temp,mask


    # # 9、图像坐标，转换为图像格式   img_png.size+2 *4  4=x,y,深度，反射强度
    # in_img_temp = np.zeros((image_size[1],image_size[0],4), dtype=np.float32)    # 图像模板
    # mask = np.zeros((image_size[1],image_size[0]), dtype=np.uint8)
    # # in_img_temp = np.zeros((image_size[1],image_size[0],4), dtype=np.float32) + 10   # 图像模板


    # box_center = (in_points_img_4[:,0].mean(),in_points_img_4[:,1].mean())

    # for index_point in range(len(in_points_img_4)):
    #     point = in_points_img_4[index_point]
    #     x_index = int(point[0] + image_size[0]/2 - box_center[0] + 0.5)
    #     y_index = int(point[1] + image_size[1]/2 - box_center[1] + 0.5)
    #     try:
    #         in_img_temp[y_index,x_index] = point
    #         kernal = 5
    #         zero = kernal//2
    #         for i in range(kernal):
    #             for j in range(kernal):
    #                 mask[y_index-zero+i, x_index-zero+j] = 255
    #     except:
    #         pass
    #         # print([y_index,x_index])


    # # 10、图像数值范围变换
    # # /1280 /384 /100 /1
    # in_img_temp[:,:,0] = in_img_temp[:,:,0]/2000
    # in_img_temp[:,:,1] = in_img_temp[:,:,1]/1000
    # in_img_temp[:,:,2] = in_img_temp[:,:,2]/150

    # in_img_temp[in_img_temp > 1] = 1

    # return in_img_temp[:,:,:3],mask











def tensor2im(input_image):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        image_numpy = np.transpose(image_numpy, (1, 2, 0))  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy



def save_results(visuals,result_dir):
    A_path = visuals['image_path_A'][0]
    A_id = A_path.split('/')[-1].split('.')[0]

    img_real_A = Image.fromarray(((visuals['real_A'].data)[0].cpu().numpy().squeeze() * 255).astype(np.uint8))
    img_fake_B = Image.fromarray(((visuals['fake_B'].data)[0].cpu().numpy().squeeze() * 255).astype(np.uint8))
    img_rec_A = Image.fromarray(((visuals['rec_A'].data)[0].cpu().numpy().squeeze() * 255).astype(np.uint8))
    img_idt_B = Image.fromarray(((visuals['idt_B'].data)[0].cpu().numpy().squeeze() * 255).astype(np.uint8))
    img_mask_A = Image.fromarray((visuals['A_mask'][0].cpu().numpy().astype(np.uint8).squeeze()) * 255)

    # img_real_A = (img_real_A * 255.0).astype(np.uint8)
    # img_fake_B = (img_fake_B * 255.0).astype(np.uint8)
    # img_rec_A = (img_rec_A * 255.0).astype(np.uint8)
    # img_idt_B = (img_idt_B * 255.0).astype(np.uint8)

    # img_real_A = Image.fromarray(img_real_A)
    # img_fake_B = Image.fromarray(img_fake_B)
    # img_rec_A = Image.fromarray(img_rec_A)
    # img_idt_B = Image.fromarray(img_idt_B)

    img_real_A_path = '{}/{}_real_A.png'.format(result_dir,A_id)
    img_fake_B_path = '{}/{}_fake_B.png'.format(result_dir,A_id)
    img_rec_A_path = '{}/{}_rec_A.png'.format(result_dir,A_id)
    img_idt_B_path = '{}/{}_idt_B.png'.format(result_dir,A_id)
    img_mask_A_path = '{}/{}_mask_A.png'.format(result_dir,A_id)

    img_real_A.save(img_real_A_path)
    img_fake_B.save(img_fake_B_path)
    img_rec_A.save(img_rec_A_path)
    img_idt_B.save(img_idt_B_path)
    img_mask_A.save(img_mask_A_path)


    B_path = visuals['image_path_B'][0]
    B_id = B_path.split('/')[-1].split('.')[0]

    img_real_B = Image.fromarray(((visuals['real_B'].data)[0].cpu().numpy().squeeze() * 255).astype(np.uint8))
    img_fake_A = Image.fromarray(((visuals['fake_A'].data)[0].cpu().numpy().squeeze() * 255).astype(np.uint8))
    img_rec_B = Image.fromarray(((visuals['rec_B'].data)[0].cpu().numpy().squeeze() * 255).astype(np.uint8))
    img_idt_A = Image.fromarray(((visuals['idt_A'].data)[0].cpu().numpy().squeeze() * 255).astype(np.uint8))
    img_mask_B = Image.fromarray((visuals['B_mask'][0].cpu().numpy().astype(np.uint8).squeeze()) * 255)

    # img_real_B = (img_real_B * 255.0).astype(np.uint8)
    # img_fake_A = (img_fake_A * 255.0).astype(np.uint8)
    # img_rec_B = (img_rec_B * 255.0).astype(np.uint8)
    # img_idt_A = (img_idt_A * 255.0).astype(np.uint8)

    # img_real_B = Image.fromarray(img_real_B)
    # img_fake_A = Image.fromarray(img_fake_A)
    # img_rec_B = Image.fromarray(img_rec_B)
    # img_idt_A = Image.fromarray(img_idt_A)

    img_real_B_path = '{}/{}_real_B.png'.format(result_dir,B_id)
    img_fake_A_path = '{}/{}_fake_A.png'.format(result_dir,B_id)
    img_rec_B_path = '{}/{}_rec_B.png'.format(result_dir,B_id)
    img_idt_A_path = '{}/{}_idt_A.png'.format(result_dir,B_id)
    img_mask_B_path = '{}/{}_mask_B.png'.format(result_dir, B_id)

    img_real_B.save(img_real_B_path)
    img_fake_A.save(img_fake_A_path)
    img_rec_B.save(img_rec_B_path)
    img_idt_A.save(img_idt_A_path)
    img_mask_B.save(img_mask_B_path)


    # visual_names_A = ['real_A', 'fake_B', 'rec_A']
    # visual_names_B = ['real_B', 'fake_A', 'rec_B']
    # visual_names_A.append('idt_B')
    # visual_names_B.append('idt_A')


if __name__ == '__main__':

    import os
    from PIL import Image

    def make_dataset2(dir, max_dataset_size=float("inf")):
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                path = os.path.join(root, fname)
                images.append(path)
        return images[:min(max_dataset_size, len(images))]



    A_paths = sorted(make_dataset2('/code/mix-pe/cycle_gan/trainA'))
    B_paths = sorted(make_dataset2('/code/mix-pe/cycle_gan/trainB'))

    dataroot = '/code/mix-pe/cycle_gan'


    for index in range(len(A_paths)):
    # for index in range(2):
        A_path = A_paths[index % len(A_paths)]  # make sure index is within then range


        img = obj_img(A_path, dataroot)

        img = img * 255 + 10
        img = img.astype(np.uint8)
        img = Image.fromarray(img)
        img.save('/code/mix-pe/cycle_gan/resultsA/{}.png'.format(A_path.split('/')[-1].split('.')[0]))

    for index in range(len(B_paths)):
    # for index in range(2):
        B_path = B_paths[index % len(B_paths)]  # make sure index is within then range


        img = obj_img(B_path, dataroot)

        img = img * 255 + 10
        img = img.astype(np.uint8)
        img = Image.fromarray(img)
        img.save('/code/mix-pe/cycle_gan/resultsB/{}.png'.format(B_path.split('/')[-1].split('.')[0]))