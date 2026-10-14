# --robot.cameras="{ 'wrist': {'type': 'opencv', 'index_or_path': 0, 'width': 640, 'height': 360, 'fps': 30}}" `
lerobot-teleoperate `
    --robot.type=so101_follower `
    --robot.port=COM6 `
    --robot.id=LXFA0001 `
	--robot.cameras="{'wrist': {'type':'opencv', 'index_or_path':0, 'width':640, 'height':360, 'fps':30}, 'front': {'type':'opencv', 'index_or_path':1, 'width':640, 'height':360, 'fps':30}}" `
    --teleop.type=so101_leader `
    --teleop.port=COM5 `
    --teleop.id=LXLA0001 `
	--display_data=true