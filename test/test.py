def process_string(s):
    result = []
    for char in s:
        if char.isdigit():
            result.append(str(int(char)))
        else:
            result.append(char)
    return ",".join(result)


text = "0123456789aáảàãạâấẩầẫậăắẳằẵặbcdđeéẻèẽẹêếểềễệfghiíỉìĩịjklmnoóỏòõọôốổồỗộơớởờỡợpqrstuúủùũụưứửừữựvwxyýỷỳỹỵz,.:"
print(process_string(text))
