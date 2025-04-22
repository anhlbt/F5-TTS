import re
import os
import shutil
import sys

# from vfastpunct import VFastPunct

break_words = [
    "không những mà còn",
    "chẳng hạn như",
    "ví dụ như",
    "nói cách khác",
    "cụ thể là",
    "không chỉ mà còn",
    "mặt khác",
    "ngoài ra",
    "hơn nữa",
    "trái lại",
    "ngược lại",
    "trong khi",
    "dẫu cho",
    "dù cho",
    "mặc cho",
    "mặc kệ",
    "miễn là",
    "miễn sao",
    "chỉ cần",
    "đối với",
    "như thể",
    "theo đó",
    "theo như",
    "cùng lúc đó",
    "cùng lúc",
    "thậm chí",
    "để cho",
    "để",
    "vì vậy",
    "bởi vì",
    "vì thế",
    "do đó",
    "do vậy",
    "nếu như",
    "hay",
    "vì",
    "nên",
    "và",
    "nhưng",
    "rồi",
    "lại",
    "cũng",
    "vẫn",
    "như",
    "nếu",
    "khi",
    "lúc",
    "bởi",
    "do",
]


def chunk_text(text, max_chars=115):
    # Define length limits
    threshold = int(max_chars * 0.8)  # 80% của max_chars
    threshold_2 = int(max_chars * 1.2)  # 120% của max_chars
    punctuation_marks = "。？！，、；：”’》」』）】…—"
    english_punctuation = ".?!,:;)}…"
    final_punctuation = ".?"  # Chỉ dùng các dấu này để cắt khi vượt ngưỡng 80%
    # Results list
    result = []
    # Starting location
    pos = 0
    text = text.strip()
    text_length = len(text)

    i = 0
    last_punctuation_pos = None
    last_space_pos = None

    while i < text_length:
        char = text[i]
        current_length = i - pos + 1

        # Ghi nhớ vị trí dấu câu và dấu trắng
        if char in punctuation_marks or char in english_punctuation:
            if char == "." and i < text_length - 1 and re.match(r"\d", text[i + 1]):
                i += 1
                continue
            last_punctuation_pos = i
        elif char.isspace():
            last_space_pos = i

        # Trường hợp đặc biệt: cho phép vượt max_chars nếu trong threshold_2 và kết thúc bằng "."
        if (
            current_length <= threshold_2
            and char == "."
            and (i == text_length - 1 or text[i + 1].isspace())
        ):
            result.append(text[pos : i + 1].strip())
            pos = i + 1
            i = pos
            last_punctuation_pos = None
            last_space_pos = None
            continue

        # Kiểm tra khi vượt quá max_chars
        if current_length > max_chars:
            # Nếu vượt threshold_2 thì không áp dụng trường hợp đặc biệt
            if current_length > threshold_2:
                # Ưu tiên cắt ở dấu câu trước max_chars
                if (
                    last_punctuation_pos is not None
                    and last_punctuation_pos - pos + 1 <= max_chars
                ):
                    result.append(text[pos : last_punctuation_pos + 1].strip())
                    pos = last_punctuation_pos + 1
                # Nếu không có dấu câu, cắt ở dấu trắng
                elif (
                    last_space_pos is not None and last_space_pos - pos + 1 <= max_chars
                ):
                    result.append(text[pos:last_space_pos].strip())
                    pos = last_space_pos + 1
                # Nếu không có điểm cắt nào, tìm dấu trắng gần nhất trước max_chars
                else:
                    chunk = text[pos : pos + max_chars]
                    last_space = chunk.rfind(" ")
                    if last_space != -1:
                        result.append(chunk[:last_space].strip())
                        pos = pos + last_space + 1
                    else:
                        # Nếu không có dấu trắng nào trong max_chars, cắt ở dấu trắng cuối cùng trước đó
                        if pos > 0:
                            prev_chunk = text[:pos]
                            last_space_before = prev_chunk.rfind(" ")
                            if last_space_before != -1:
                                result[-1] = prev_chunk[:last_space_before].strip()
                                pos = last_space_before + 1
                            else:
                                pos += max_chars
                        else:
                            result.append(chunk.strip())
                            pos += max_chars
                i = pos
                last_punctuation_pos = None
                last_space_pos = None
            # Nếu trong threshold_2, tiếp tục để kiểm tra dấu chấm
        # Cắt ở ngưỡng 80% với dấu câu ưu tiên
        elif current_length >= threshold and char in final_punctuation:
            result.append(text[pos : i + 1].strip())
            pos = i + 1
            i = pos
            last_punctuation_pos = None
            last_space_pos = None

        i += 1

    # Xử lý phần text còn lại
    if pos < text_length:
        remaining = text[pos:].strip()
        while remaining:
            if len(remaining) <= max_chars or (
                len(remaining) <= threshold_2 and remaining.endswith(".")
            ):
                result.append(remaining)
                break

            # Tìm điểm cắt cho phần còn lại
            last_punctuation_pos = None
            last_space_pos = None
            for j, char in enumerate(remaining[:max_chars]):
                if char in punctuation_marks or char in english_punctuation:
                    if (
                        char == "."
                        and j < len(remaining) - 1
                        and re.match(r"\d", remaining[j + 1])
                    ):
                        continue
                    last_punctuation_pos = j
                elif char.isspace():
                    last_space_pos = j

            if last_punctuation_pos is not None:
                result.append(remaining[: last_punctuation_pos + 1].strip())
                remaining = remaining[last_punctuation_pos + 1 :].strip()
            elif last_space_pos is not None:
                result.append(remaining[:last_space_pos].strip())
                remaining = remaining[last_space_pos + 1 :].strip()
            else:
                chunk = remaining[:max_chars]
                last_space = chunk.rfind(" ")
                if last_space != -1:
                    result.append(chunk[:last_space].strip())
                    remaining = remaining[last_space + 1 :].strip()
                else:
                    if result:
                        last_chunk = result.pop()
                        last_space_before = last_chunk.rfind(" ")
                        if last_space_before != -1:
                            result.append(last_chunk[:last_space_before].strip())
                            remaining = last_chunk[last_space_before + 1 :] + remaining
                        else:
                            result.append(last_chunk)
                            result.append(chunk.strip())
                            remaining = remaining[max_chars:].strip()
                    else:
                        result.append(chunk.strip())
                        remaining = remaining[max_chars:].strip()

    return result


# Kiểm tra với đoạn text của bạn
text = """
thông điệp mâu thuẫn và sự hỗn loạn của thị trường. sự thay đổi liên tục trong quan điểm của chính tổng thống trump về mục đích của chính sách thuế quan mới đang làm đảo lộn nỗ lực của các quan chức kinh tế cấp cao của ông trong việc truyền đạt một thông điệp thống nhất rằng thuế đối ứng sẽ có hiệu lực vào ngày chín tháng bốn mà không có bất cứ thay đổi nào. cả đồng minh và những người phản đối ông trump đều nhấn mạnh cần phải có sự thống nhất ở thời điểm quan trọng hiện nay. việc thiếu tính thống nhất trong việc truyền đạt thông điệp sẽ không giải quyết được mối đe dọa lớn khác đang hiện hữu. các đòn trả đũa. trong khi tổng thống trump liên tục gửi đi nhiều thông điệp khác nhau, một cố vấn cho biết ông đã chủ động không phát biểu công khai hay trả lời câu hỏi trước ống kính khi thị trường giao dịch vẫn mở cửa trong ngày bốn tháng bốn. mặc dù nhà trắng tiếp tục bảo vệ mạnh mẽ chiến lược thuế quan của tổng thống, nhưng họ không ngăn được sự bất mãn và tức giận ngày càng dâng cao trong phản ứng đồng loạt trên toàn cầu. chứng khoán mỹ đã giảm điểm trong ngày thứ hai liên tiếp sau khi trung quốc tuyên bố rằng sẽ áp thuế ba mươi tư phần trăm đối với hàng hóa nhập khẩu từ mỹ. chỉ số dow jones kết thúc ngày giao dịch giảm hơn mười phần trăm so với mức cao kỷ lục vào tháng mười hai năm hai nghìn không trăm mười bốn. tổng thống không bị chi phối bởi thông tin từ các thị trường, ông theo dõi các thị trường như những người khác, một cố vấn chính trị của ông trump nói với xê nờ nờ. một người từng trò chuyện với ông trump trong khi thị trường đang giảm giá trong ngày ba tháng bốn cho biết, tổng thống tỏ ra khá bình tĩnh về kế hoạch của mình, cho rằng thuế quan chỉ là một phần trong một chiến lược kinh tế rộng lớn hơn vẫn đang hình thành. tuy nhiên, một người khác nói rằng mức độ chấp nhận của ông về việc các thị trường sụt giảm đang tới gần giới hạn. nhà trắng đã nhận được nhiều cuộc gọi từ các chủ doanh nghiệp và các nhóm vận động, nhưng không rõ trump có hiểu được mức độ phản ứng tiêu cực và liệu điều này có ảnh hưởng đến lập trường của ông trong những ngày trước khi thuế đối ứng có hiệu lực vào ngày chín tháng bốn hay không. phản ứng của chủ tịch fed và giới doanh nghiệp mỹ. phát biểu tại một sự kiện ở arlington, virginia, ngày bốn tháng bốn, chủ tịch cục dự trữ liên bang fed jerome powell, nói rằng ngân hàng trung ương đã bị bất ngờ trước quy mô đòn thuế quan của trump. chúng ta đang đối mặt với một triển vọng rất không chắc chắn với rủi ro cao cả về tỷ lệ thất nghiệp và lạm phát. mặc dù thuế quan sẽ tạo ra ít nhất một sự gia tăng tạm thời về lạm phát, nhưng nó cũng có thể có tác động kéo dài, ông powell cảnh báo. trong khi đó, giới doanh nghiệp mỹ đang rất tức giận. theo các cuộc trò chuyện với một số giám đốc điều hành, bộ trưởng tài chính scott bessent đã nhận được nhiều cuộc gọi giận dữ từ các lãnh đạo doanh nghiệp, một số người trong số họ đang cân nhắc kiện chính quyền về chính sách thuế mới và cả tình trạng khẩn cấp quốc gia mà ông trump đưa ra làm lý do cho các biện pháp này. một giám đốc điều hành doanh nghiệp ở washington dê xê cho biết phạm vi và mức độ mạnh tay trong chính sách thuế quan của tổng thống đã gây sốc, đặc biệt là khi xét đến số lượng công ty đã phải điều chỉnh công việc làm ăn kinh doanh của họ cho phù hợp với các mục tiêu chính sách mà ông trump cho là đúng đắn. mặc dù có sự phản đối từ nhiều thành viên trong nội các của ông về việc không cần thiết phải khiến các đồng minh phẫn nộ và làm suy giảm các thị trường toàn cầu, quan điểm của trump về thuế quan vẫn rất vững chắc. về vấn đề này, ông ấy dường như sẽ không thay đổi, cố vấn thương mại của tổng thống trump, peter navarro cho biết. trên phố wall, chủ đề được thảo luận phổ biến hiện nay là có nên công khai lên tiếng chống lại chính sách mới của tổng thống hay không. các giám đốc điều hành trong khu vực tư nhân cũng đang tranh luận về việc có nên thuê một nhà vận động hành lang thân thiện với ông trump để cố gắng tìm kiếm một ngoại lệ từ chính sách này hay không.
"""
chunks = chunk_text(text, max_chars=150)
for idx, chunk in enumerate(chunks):
    print(f"gen_text {idx} ({len(chunk)}): {chunk}")
