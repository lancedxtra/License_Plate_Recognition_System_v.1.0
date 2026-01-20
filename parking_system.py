import datetime

class ParkingManager:
    def __init__(self, capacity=100, rate_per_hour=10):
        self.capacity = capacity
        self.rate_per_hour = rate_per_hour
        self.records = {}  # 存储格式: {plate_number: entry_time}

    def entry(self, plate_number):
        """
        处理车辆入场
        :param plate_number: 车牌号 (str)
        :return: (bool, msg) 成功状态和提示信息
        """
        if plate_number in self.records:
            return False, f"车辆 {plate_number} 已在场内"
        if len(self.records) >= self.capacity:
            return False, "车位已满"
        
        self.records[plate_number] = datetime.datetime.now()
        return True, f"车辆 {plate_number} 入场成功"

    def exit(self, plate_number):
        """
        处理车辆离场并计算费用
        :param plate_number: 车牌号 (str)
        :return: (bool, dict/str) 成功时返回详细账单，失败返回错误信息
        """
        if plate_number not in self.records:
            return False, "未找到该车辆记录"

        entry_time = self.records.pop(plate_number)
        exit_time = datetime.datetime.now()
        
        # 计算时长
        duration = exit_time - entry_time
        hours = duration.total_seconds() / 3600
        # 计费逻辑：不满1小时按1小时计，之后按比例
        cost = round(max(1, hours) * self.rate_per_hour, 2)

        receipt = {
            "plate_number": plate_number,
            "entry_time": entry_time.strftime("%Y-%m-%d %H:%M:%S"),
            "exit_time": exit_time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration": str(duration).split('.')[0],
            "cost": cost
        }
        return True, receipt

    def get_available_spots(self):
        """获取剩余车位数量"""
        return self.capacity - len(self.records)