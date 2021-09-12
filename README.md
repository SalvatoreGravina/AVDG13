# AVDG13

The project include 3 tasks:
- [predict the state of a traffic light] (https://youtu.be/eHCn3hBHzhY)
- [avoid collision with objects] (https://youtu.be/peMbqIoSts8)
- [avoid collision with vehicles] (https://youtu.be/zgoITMXW6A4)


## Authors: Group 13

| Name | Student ID |
|--------------|--------|
|Vincenzo di Somma | 0622701283|
|Salvatore Gravina | 0622701063|
|Ferdinando Guarino | 0622701321|

## REQUIREMENTS

- A ***requirements.txt*** file is provided with required packages.
- Compatible with CARLA 0.8.4 and Python 3.6.4.
- Clone this repository in the ***PythonClient*** folder.

## SETUP

1. Access CARLA root folder
2. Start CARLA server on map **TOWN 01**

```cmd
start ./CarlaUE4.exe /Game/Maps/Town01 -carla-server -windowed -fps=30 -quality-level=Epic -benchmark
```

3. Set ***CONFIGURABLE PARAMETERS*** in main.py

4. Start client

```cmd
python ./PythonClient/AVDG13/main.py
```
