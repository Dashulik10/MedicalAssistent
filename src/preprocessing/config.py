
# ==============================================================================
# 🎯 БЫСТРЫЕ ПРЕСЕТЫ
# ==============================================================================

def get_preset(name: str):
    """
    Готовые наборы параметров для разных случаев
    
    Использование:
        В image_preprocessor.py в __main__ измените на:
        from config import get_preset
        params = get_preset("medical_llm")
    """
    presets = {
        # ЛУЧШИЙ ВАРИАНТ для медицинских отчетов + LLM (БЕЗ бинаризации!)
        "medical_llm": {
            "max_dimension": 3000,
            "background_kernel_divisor": 35,
            "background_weight": 0.6,
            "median_blur_size": 3,
            "bilateral_filter": False,
            "bilateral_d": 9,
            "bilateral_sigma_color": 75,
            "bilateral_sigma_space": 75,
            "clahe_clip_limit": 2.8,
            "clahe_tile_size": 8,
            "sharpen_amount": 1.6,
            "sharpen_blur_sigma": 1.0,
            "use_binarization": False,
            "binarization_block_size": 51,
            "binarization_c": 5,
            "contrast_enhancement": True,
            "gamma_correction": 1.0,
        },
        
        # МАКСИМАЛЬНАЯ ЧЕТКОСТЬ для медицинских отчетов + LLM
        "medical_llm_sharp": {
            "max_dimension": 3000,
            "background_kernel_divisor": 35,
            "background_weight": 0.65,
            "median_blur_size": 3,
            "bilateral_filter": False,
            "bilateral_d": 9,
            "bilateral_sigma_color": 75,
            "bilateral_sigma_space": 75,
            "clahe_clip_limit": 3.2,  # Более высокий контраст
            "clahe_tile_size": 8,
            "sharpen_amount": 2.0,  # Более высокая резкость
            "sharpen_blur_sigma": 0.9,
            "use_binarization": False,
            "binarization_block_size": 51,
            "binarization_c": 5,
            "contrast_enhancement": True,
            "gamma_correction": 1.0,
        },
        
        # Для медицинских отчетов + традиционный OCR (С бинаризацией)
        "medical_ocr": {
            "max_dimension": 3500,
            "background_kernel_divisor": 25,
            "background_weight": 0.9,
            "median_blur_size": 3,
            "bilateral_filter": False,
            "clahe_clip_limit": 4.0,
            "clahe_tile_size": 8,
            "sharpen_amount": 2.0,
            "sharpen_blur_sigma": 0.9,
            "use_binarization": True,
            "binarization_block_size": 45,
            "binarization_c": 8,
            "contrast_enhancement": True,
            "gamma_correction": 1.0,
        },
        
        # Для очень размытых/блеклых изображений
        "aggressive": {
            "max_dimension": 3500,
            "background_kernel_divisor": 25,
            "background_weight": 0.9,
            "median_blur_size": 3,
            "bilateral_filter": True,
            "bilateral_d": 3,
            "bilateral_sigma_color": 75,
            "bilateral_sigma_space": 75,
            "clahe_clip_limit": 4.5,
            "clahe_tile_size": 8,
            "sharpen_amount": 2.2,
            "sharpen_blur_sigma": 0.8,
            "use_binarization": False,
            "binarization_block_size": 41,
            "binarization_c": 12,
            "contrast_enhancement": True,
            "gamma_correction": 0.9,
        },
        
        # Для изображений среднего качества
        "balanced": {
            "max_dimension": 3000,
            "background_kernel_divisor": 30,
            "background_weight": 0.7,
            "median_blur_size": 3,
            "bilateral_filter": False,
            "clahe_clip_limit": 3.0,
            "clahe_tile_size": 8,
            "sharpen_amount": 1.5,
            "sharpen_blur_sigma": 1.0,
            "use_binarization": False,
            "binarization_block_size": 51,
            "binarization_c": 5,
            "contrast_enhancement": True,
            "gamma_correction": 1.0,
        },
        
        # Для уже хороших изображений (легкая обработка)
        "gentle": {
            "max_dimension": 3000,
            "background_kernel_divisor": 40,
            "background_weight": 0.5,
            "median_blur_size": 3,
            "bilateral_filter": False,
            "clahe_clip_limit": 2.0,
            "clahe_tile_size": 12,
            "sharpen_amount": 1.3,
            "sharpen_blur_sigma": 1.2,
            "use_binarization": False,
            "binarization_block_size": 61,
            "binarization_c": 3,
            "contrast_enhancement": True,
            "gamma_correction": 1.0,
        },
    }
    
    return presets.get(name, presets["medical_llm"])


