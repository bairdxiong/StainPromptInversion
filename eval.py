import argparse
from evaluate_metric.eval_score import fid_psnr_css_ssim_msssim

def main():
    paser=argparse.ArgumentParser(description="Evaluate Results Metric")
    paser.add_argument("--source_path",type=str,required=True,help="The dir path of source image")
    paser.add_argument("--translate_path",type=str,required=True,help="The dir path of translated image(RESULTS)")
    paser.add_argument("--style_gt_path",type=str,required=True,help="The dir path of groundtruth style image(ANHIR_val)")
    args=paser.parse_args()
    fid_psnr_css_ssim_msssim(translate_path=args.translate_path,
                             source_path=args.source_path,gt_path=args.style_gt_path)
if __name__ == "__main__":
    main()
