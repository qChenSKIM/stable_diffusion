# Stable diffusion on EC2 instance

## why use EC2 as sever?

Stable diffusion model is very big to download locally, it requires large storage space.

## steps:
0. in huggingface, get your token, create `.env` file with `HG_TOKEN=your_huggingface_token`, don't forget to get access from [stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4)
1. create a git repo and upload to github
2. start EC2 instance with ubuntu system (enable ssh, allow all traffic into this instance)
3. ssh into EC2 instance
4. install python3 (installed by default `Python3 --version`)
5. git clone `git clone https://github.com/qChenSKIM/stable_diffusion.git`
6. `cd stable_diffusion & pip3 install -r req.txt`
- problem with pip3, [check here](https://askubuntu.com/questions/1254309/not-installing-pip-on-ubuntu-20-04)
7. run the app: `python3 app.py`
8. check EC2 instance public IP: then enter "EC2 instance public IP:5000"


## Extra steps: Routing traffic from AWS Route 53 to EC2 instance

1. open domin53
2. create A record, route to EC2 elastic IP or public IP




