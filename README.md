# This the Pytorch Implementation of LightGODE

## How to use:
1. Install the torch-based environment
```
pip install -r requirements.txt
```

2. Get the beauty, toys-and-games, gowalla dataset under dataset folder

3. Running on different datasets:

Amazon-Beauty
```
python run_recbole.py -m LightGODE -d amazon-beauty
```

Amazon-Toys-and-Games
```
python run_recbole.py -m LightGODE -d amazon-toys-games
```

Gowalla
```
python run_recbole.py -m LightGODE -d gowalla
```