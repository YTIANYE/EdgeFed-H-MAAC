# 记录控制台日志
class PRINT_LOGS:
    def __init__(self, m_time):
        # m_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        self.m_time = m_time

    def open(self):
        logs = open('logs/print_logs/%s.txt' % self.m_time, 'a')  # 'w'覆盖 'a'追加
        return logs
