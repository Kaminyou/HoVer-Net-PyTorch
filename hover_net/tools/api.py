import matplotlib.pyplot as plt

from hover_net.dataloader import get_dataloader
from hover_net.postprocess import process
from hover_net.process import infer_step, visualize_instances_dict
from hover_net.tools.coco import parse_single_instance


def infer_one_image(
    image_path,
    model,
    nr_types=3,
    input_size=(512, 512),
    device="cuda",
    show=False
):
    inference_dataloader = get_dataloader(
        data_path=[image_path],
        input_shape=input_size,
        run_mode="inference_single"
    )

    detection_list = []
    segmentation_list = []
    for step_idx, data in enumerate(inference_dataloader):
        assert data.shape[0] == 1

        test_result_output = infer_step(
            batch_data=data, model=model, device=device
        )
        image_id = 0
        for curr_image_idx in range(len(test_result_output)):
            pred_inst, inst_info_dict = process(
                test_result_output[curr_image_idx],
                nr_types=nr_types,
                return_centroids=True
            )

            for single_inst_info in inst_info_dict.values():
                detection_dict, segmentation_dict = parse_single_instance(
                    image_id, single_inst_info
                )
                detection_list.append(detection_dict)
                segmentation_list.append(segmentation_dict)

            if show:
                src_image = data[0].numpy()
                type_info_dict = {
                    "0": ["nolabe", [0, 0, 0]],
                    "1": ["neopla", [255, 0, 0]],
                    "2": ["inflam", [0, 255, 0]],
                    "3": ["connec", [0, 0, 255]],
                    "4": ["necros", [255, 255, 0]],
                    "5": ["no-neo", [255, 165, 0]]
                }
                type_info_dict = {
                    int(k): (
                        v[0], tuple(v[1])
                    ) for k, v in type_info_dict.items()
                }
                overlay_kwargs = {
                    "draw_dot": True,
                    "type_colour": type_info_dict,
                    "line_thickness": 2,
                }
                overlaid_img = visualize_instances_dict(
                    src_image.copy(), inst_info_dict, **overlay_kwargs
                )
                plt.imshow(overlaid_img)
                plt.axis("off")
                plt.show()

    return inst_info_dict, detection_list, segmentation_list
