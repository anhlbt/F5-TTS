import re
from underthesea import chunk as chunk_function

from underthesea import word_tokenize

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
    threshold_2 = int(max_chars * 1.2)  # 120% của max_chars - độ dài tối đa cho chunk
    min_chunk_length = int(max_chars * 0.2)  # Ngưỡng tối thiểu để xem xét chunk
    final_punctuation = ".?!"  # Dấu câu ưu tiên để kết thúc chunk
    english_punctuation = ".?!,:;)}…"  # Dấu câu tiếng Anh để kiểm tra chunk tạm
    punctuation_marks = (
        "。？！，、；：”’》」』）】…—"  # Dấu câu khác để kiểm tra chunk tạm
    )

    # Results list
    result = []
    # Starting position
    pos = 0
    text = text.strip()
    text_length = len(text)

    while pos < text_length:
        chunk_temp = None
        last_valid_end = None

        # Duyệt từ vị trí hiện tại để tìm điểm cắt tối ưu
        for i in range(pos, min(pos + threshold_2, text_length)):
            current_length = i - pos + 1
            char = text[i]

            # Ưu tiên 1: Kết thúc bằng final_punctuation, dài nhất có thể (tối đa threshold_2)
            if (
                char in final_punctuation
                and current_length <= threshold_2
                and current_length > min_chunk_length
            ):
                last_valid_end = i
                if (
                    i == text_length - 1 or text[i + 1].isspace()
                ):  # Đảm bảo không cắt giữa từ
                    chunk_temp = text[pos : i + 1].strip()
                    break

            # Ưu tiên 2: Lưu chunk tạm nếu thoả min_chunk_length < length < max_chars
            # và kết thúc bằng dấu trong english_punctuation hoặc punctuation_marks
            elif (
                min_chunk_length < current_length <= max_chars
                and char in (english_punctuation + punctuation_marks)
                and (i == text_length - 1 or text[i + 1].isspace())
            ):
                if not chunk_temp:  # Chỉ lưu chunk_temp đầu tiên thoả mãn
                    chunk_temp = text[pos : i + 1].strip()

        # Xử lý chunk đã tìm được
        if last_valid_end is not None:  # Ưu tiên 1: Có final_punctuation
            new_chunk = text[pos : last_valid_end + 1].strip()
            result.append(new_chunk)
            pos = last_valid_end + 1
        elif chunk_temp:  # Ưu tiên 2: Có chunk tạm thoả mãn
            result.append(chunk_temp)
            pos += len(chunk_temp)
        else:  # Ưu tiên 3: Không có dấu, cắt dựa trên penultimate word
            chunk = text[pos : pos + max_chars]
            last_space = chunk.rfind(" ")
            if last_space != -1:
                sub_chunk = text[pos : pos + last_space]
                tokenized_words = word_tokenize(sub_chunk)
                if len(tokenized_words) >= 2:
                    penultimate_word = tokenized_words[-2]
                    cut_pos = sub_chunk.rfind(penultimate_word) + len(penultimate_word)
                    new_chunk = text[pos : pos + cut_pos].strip()
                    result.append(new_chunk)
                    pos = pos + cut_pos
                else:
                    new_chunk = sub_chunk.strip()
                    result.append(new_chunk)
                    pos = pos + last_space
            else:  # Nếu không có khoảng trắng, cắt cứng ở max_chars
                new_chunk = chunk.strip()
                result.append(new_chunk)
                pos += max_chars

        # Bỏ qua khoảng trắng sau chunk
        while pos < text_length and text[pos].isspace():
            pos += 1

    return result


# Kiểm tra với đoạn text của bạn
text = """
thông điệp mâu thuẫn và sự hỗn loạn của thị trường. sự thay đổi liên tục trong quan điểm của chính tổng thống trump về mục đích của chính sách thuế quan mới đang làm đảo lộn nỗ lực của các quan chức kinh tế cấp cao của ông trong việc truyền đạt một thông điệp thống nhất rằng thuế đối ứng sẽ có hiệu lực vào ngày chín tháng bốn mà không có bất cứ thay đổi nào. cả đồng minh và những người phản đối ông trump đều nhấn mạnh cần phải có sự thống nhất ở thời điểm quan trọng hiện nay. việc thiếu tính thống nhất trong việc truyền đạt thông điệp sẽ không giải quyết được mối đe dọa lớn khác đang hiện hữu. các đòn trả đũa. trong khi tổng thống trump liên tục gửi đi nhiều thông điệp khác nhau, một cố vấn cho biết ông đã chủ động không phát biểu công khai hay trả lời câu hỏi trước ống kính khi thị trường giao dịch vẫn mở cửa trong ngày bốn tháng bốn. mặc dù nhà trắng tiếp tục bảo vệ mạnh mẽ chiến lược thuế quan của tổng thống, nhưng họ không ngăn được sự bất mãn và tức giận ngày càng dâng cao trong phản ứng đồng loạt trên toàn cầu. chứng khoán mỹ đã giảm điểm trong ngày thứ hai liên tiếp sau khi trung quốc tuyên bố rằng sẽ áp thuế ba mươi tư phần trăm đối với hàng hóa nhập khẩu từ mỹ. chỉ số dow jones kết thúc ngày giao dịch giảm hơn mười phần trăm so với mức cao kỷ lục vào tháng mười hai năm hai nghìn không trăm mười bốn. tổng thống không bị chi phối bởi thông tin từ các thị trường, ông theo dõi các thị trường như những người khác, một cố vấn chính trị của ông trump nói với xê nờ nờ. một người từng trò chuyện với ông trump trong khi thị trường đang giảm giá trong ngày ba tháng bốn cho biết, tổng thống tỏ ra khá bình tĩnh về kế hoạch của mình, cho rằng thuế quan chỉ là một phần trong một chiến lược kinh tế rộng lớn hơn vẫn đang hình thành. tuy nhiên, một người khác nói rằng mức độ chấp nhận của ông về việc các thị trường sụt giảm đang tới gần giới hạn. nhà trắng đã nhận được nhiều cuộc gọi từ các chủ doanh nghiệp và các nhóm vận động, nhưng không rõ trump có hiểu được mức độ phản ứng tiêu cực và liệu điều này có ảnh hưởng đến lập trường của ông trong những ngày trước khi thuế đối ứng có hiệu lực vào ngày chín tháng bốn hay không. phản ứng của chủ tịch fed và giới doanh nghiệp mỹ. phát biểu tại một sự kiện ở arlington, virginia, ngày bốn tháng bốn, chủ tịch cục dự trữ liên bang fed jerome powell, nói rằng ngân hàng trung ương đã bị bất ngờ trước quy mô đòn thuế quan của trump. chúng ta đang đối mặt với một triển vọng rất không chắc chắn với rủi ro cao cả về tỷ lệ thất nghiệp và lạm phát. mặc dù thuế quan sẽ tạo ra ít nhất một sự gia tăng tạm thời về lạm phát, nhưng nó cũng có thể có tác động kéo dài, ông powell cảnh báo. trong khi đó, giới doanh nghiệp mỹ đang rất tức giận. theo các cuộc trò chuyện với một số giám đốc điều hành, bộ trưởng tài chính scott bessent đã nhận được nhiều cuộc gọi giận dữ từ các lãnh đạo doanh nghiệp, một số người trong số họ đang cân nhắc kiện chính quyền về chính sách thuế mới và cả tình trạng khẩn cấp quốc gia mà ông trump đưa ra làm lý do cho các biện pháp này. một giám đốc điều hành doanh nghiệp ở washington dê xê cho biết phạm vi và mức độ mạnh tay trong chính sách thuế quan của tổng thống đã gây sốc, đặc biệt là khi xét đến số lượng công ty đã phải điều chỉnh công việc làm ăn kinh doanh của họ cho phù hợp với các mục tiêu chính sách mà ông trump cho là đúng đắn. mặc dù có sự phản đối từ nhiều thành viên trong nội các của ông về việc không cần thiết phải khiến các đồng minh phẫn nộ và làm suy giảm các thị trường toàn cầu, quan điểm của trump về thuế quan vẫn rất vững chắc. về vấn đề này, ông ấy dường như sẽ không thay đổi, cố vấn thương mại của tổng thống trump, peter navarro cho biết. trên phố wall, chủ đề được thảo luận phổ biến hiện nay là có nên công khai lên tiếng chống lại chính sách mới của tổng thống hay không. các giám đốc điều hành trong khu vực tư nhân cũng đang tranh luận về việc có nên thuê một nhà vận động hành lang thân thiện với ông trump để cố gắng tìm kiếm một ngoại lệ từ chính sách này hay không.
"""

# text = """hiện tại, anh được biết đến nhiều hơn với nickname sư tử ăn chay khi chia sẻ cuộc sống chay trường và phương pháp tập luyện của mình trên các nền tảng mạng xã hội. ngoài ra, cơ thể khỏe khoắn, lối sống lành mạnh cũng giúp anh chàng nhận được nhiều sự ủng hộ, theo dõi từ netizen. nhiều người nhận xét, các video hướng dẫn nấu ăn của anh có tiết tấu nhanh, tạo cảm giác thoải mái, dễ hiểu. định hướng video của lâm quách không chỉ tập trung vào nội dung ăn chay mà mở rộng chủ đề hơn với ba cụm từ khoá ăn xanh sống lành giảm rác thải. với chủ đề này, anh tích cực tạo những content thiên về lối sống xanh, truyền cảm hứng tích cực cho mọi người sống vì bản thân, vì môi trường và vì những người xung quanh.
# """
chunks = chunk_text(text, max_chars=130)
for idx, chunk in enumerate(chunks):
    print(f"gen_text {idx} ({len(chunk)}): {chunk}")
