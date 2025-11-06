"""
Example script demonstrating medical report data extraction.

This script shows how to use the MedicalReportExtractor to extract
structured data from medical laboratory report images.
"""
import os
from pathlib import Path
from src.extraction.data_extractor import create_extractor


def main():
    """
    Example: Extract data from a single medical report image.
    """
    # Set your API key as environment variable or pass directly
    # export OPENROUTER_API_KEY="your-api-key-here"
    
    # Path to your medical report image
    image_path = Path("data/preprocessed_images/sample_report.png")
    
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        print("Please update the path to point to a valid medical report image.")
        return
    
    # Create extractor instance
    # API key will be read from OPENROUTER_API_KEY environment variable
    try:
        extractor = create_extractor()
    except ValueError as e:
        print(f"Error: {e}")
        print("\nTo fix this, set your API key:")
        print('  export OPENROUTER_API_KEY="your-api-key-here"')
        return
    
    # Extract data from image
    print(f"Extracting data from: {image_path}")
    print("This may take 10-30 seconds...\n")
    
    result = extractor.extract(image_path)
    
    # Check if extraction was successful
    if result.success:
        print("=" * 80)
        print("✓ EXTRACTION SUCCESSFUL")
        print("=" * 80)
        
        report = result.report
        
        # Display patient information
        print("\n📋 PATIENT INFORMATION:")
        print(f"  Name: {report.patient.full_name}")
        print(f"  Age: {report.patient.age or 'N/A'}")
        print(f"  Gender: {report.patient.gender or 'N/A'}")
        print(f"  Patient ID: {report.patient.patient_id or 'N/A'}")
        
        # Display report metadata
        print("\n📄 REPORT DETAILS:")
        print(f"  Type: {report.report_type}")
        print(f"  Date: {report.report_date}")
        print(f"  Lab: {report.lab_name or 'N/A'}")
        print(f"  Doctor: {report.doctor_name or 'N/A'}")
        print(f"  Report ID: {report.report_id or 'N/A'}")
        
        # Display summary statistics
        summary = report.summary()
        print("\n📊 SUMMARY:")
        print(f"  Total Tests: {summary['total_tests']}")
        print(f"  Normal: {summary['normal_tests']}")
        print(f"  Abnormal: {summary['abnormal_tests']} ({summary['abnormal_percentage']}%)")
        
        # Display all test results
        print(f"\n🧪 TEST RESULTS ({len(report.tests)} total):")
        print("-" * 80)
        
        for i, test in enumerate(report.tests, 1):
            # Format status indicator
            if test.status.value == "normal":
                status_icon = "✓"
                status_color = ""
            elif test.status.value == "high":
                status_icon = "↑"
                status_color = ""
            elif test.status.value == "low":
                status_icon = "↓"
                status_color = ""
            else:
                status_icon = "?"
                status_color = ""
            
            # Format test output
            unit_str = f" {test.unit}" if test.unit else ""
            ref_str = f" (Ref: {test.reference_range})" if test.reference_range else ""
            notes_str = f" [{test.notes}]" if test.notes else ""
            
            print(f"{i:3d}. {status_icon} {test.name}")
            print(f"      Value: {test.value}{unit_str}{ref_str}{notes_str}")
        
        # Display abnormal tests if any
        abnormal = report.get_abnormal_tests()
        if abnormal:
            print("\n⚠️  ABNORMAL RESULTS:")
            print("-" * 80)
            for test in abnormal:
                unit_str = f" {test.unit}" if test.unit else ""
                ref_str = f" (Ref: {test.reference_range})" if test.reference_range else ""
                print(f"  • {test.name}: {test.value}{unit_str}{ref_str}")
                print(f"    Status: {test.status.value.upper()}")
        
        # Display processing metadata
        print("\n⚙️  PROCESSING INFO:")
        print(f"  Model: {result.model_used}")
        print(f"  Processing Time: {result.processing_time_seconds:.2f} seconds")
        print(f"  Extraction Time: {report.extraction_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Display conclusions if available
        if report.conclusions:
            print("\n💬 CONCLUSIONS:")
            print(f"  {report.conclusions}")
        
        print("\n" + "=" * 80)
        
        # Example: Convert to dictionary for database storage
        print("\n💾 DATABASE STORAGE:")
        print("  The MedicalReport object can be directly stored in database.")
        print("  Example using Pydantic's model_dump():")
        print(f"  report_dict = report.model_dump()")
        print(f"  # Dictionary has {len(report.model_dump())} top-level fields")
        
    else:
        print("=" * 80)
        print("✗ EXTRACTION FAILED")
        print("=" * 80)
        print(f"\nError: {result.error_message}")
        print(f"Processing Time: {result.processing_time_seconds:.2f} seconds")
        print(f"Model Used: {result.model_used}")


def batch_extraction_example():
    """
    Example: Extract data from multiple medical report images.
    """
    # Directory containing medical report images
    images_dir = Path("data/preprocessed_images")
    
    # Find all PNG images
    image_paths = list(images_dir.glob("*.png"))
    
    if not image_paths:
        print(f"No images found in {images_dir}")
        return
    
    print(f"Found {len(image_paths)} images to process")
    
    # Create extractor
    try:
        extractor = create_extractor()
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Process all images
    results = extractor.extract_batch(image_paths, verbose=True)
    
    # Summary
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Total: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        total_tests = sum(len(r.report.tests) for r in successful)
        avg_time = sum(r.processing_time_seconds for r in successful) / len(successful)
        print(f"Total tests extracted: {total_tests}")
        print(f"Average processing time: {avg_time:.2f}s")
    
    # Show failed extractions
    if failed:
        print("\nFailed extractions:")
        for result in failed:
            print(f"  • {result.error_message}")


if __name__ == "__main__":
    # Run single extraction example
    main()
    
    # Uncomment to run batch extraction example
    # batch_extraction_example()

