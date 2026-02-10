**Setup,**



D:



cd D:\\jeevi\\SpdrBot\\spdrbot3\_direct\_project



conda activate env\_isaaclab







**Plain Ground,**



python scripts/rsl\_rl/train.py --task=Template-Spdrbot3-Direct-v0 --num\_envs=500 > training\_log.txt 2>\&1



python scripts\\rsl\_rl\\play.py --task=Template-Spdrbot3-Direct-v0 --num\_env=5







**Rough Ground,**



python scripts/rsl\_rl/train.py --task Template-Spdrbot3-Rough-Direct-v0 --num\_envs=500 --resume --checkpoint "D:\\jeevi\\SpdrBot\\spdrbot3\_direct\_project\\logs\\rsl\_rl\\spdr3\\2026-02-06\_21-08-12\\model\_499.pt" > training\_log.txt 2>\&1



python scripts/rsl\_rl/play.py --task Template-Spdrbot3-Rough-Direct-v0 --num\_env=5







**Box Ground,**



python scripts/rsl\_rl/train.py --task Template-Spdrbot3-Boxes-Direct-v0 --num\_envs=500 --resume --checkpoint "D:\\jeevi\\SpdrBot\\spdrbot3\_direct\_project\\logs\\rsl\_rl\\spdr3\\2026-02-06\_21-08-12\\model\_499.pt" > training\_log.txt 2>\&1



python scripts\\rsl\_rl\\play.py --task=Template-Spdrbot3-Boxes-Direct-v0 --num\_env=5







**Manual Running,**



D:



cd D:\\jeevi\\IsaacSim



python D:\\jeevi\\SpdrBot\\spdrbot3\_direct\_project\\spyderbot\_test.py

