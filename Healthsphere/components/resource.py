import streamlit as st 
from streamlit_lottie import st_lottie
import json
import os
from components.utils import load_lottie_file

resource_animation_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Animation", "resource.json")
resource_animation = load_lottie_file(resource_animation_path)

def show_resource():
    col1, col2 = st.columns([1, 5])
    with col1:
        st_lottie(resource_animation, height=100, width=100, speed=10)

    with col2:
        st.header("Educational Resources")

    st.write("Explore important health insights through interactive resources.")

    # Expandable Sections
    with st.expander("⭕ Symptoms & Early Warning Signs"):
        st.markdown("- **[Brain Disease](https://www.apollohospitals.com/health-library/understanding-brain-tumors-types-symptoms-and-diagnosis/)**")
        st.markdown("- **[Heart Disease](https://www.maxhealthcare.in/blogs/heart-disease-symptoms-and-causes/)**")
        st.markdown("- **[Kidney Disease](https://www.medanta.org/patient-education-blog/kidney-disease-know-the-symptoms-types-and-causes/)**")
        st.markdown("- **[Liver Disease](https://www.manipalhospitals.com/blog/15-symptoms-of-serious-liver-disease/)**")
        st.markdown("- **[Diabetes](https://www.onetouchdiabetes.co.in/facts-about-diabetes/start-your-journey/signs-and-symptoms-of-diabetes)**")
        st.markdown("- **[Parkinson's Disease](https://pmc.ncbi.nlm.nih.gov/articles/PMC9764889/)**")

    with st.expander("⭕ Risk Factors"):
        st.markdown("- **[Brain Disease](https://www.maxhealthcare.in/blogs/decoding-brain-tumour-and-its-cure/)**")
        st.markdown("- **[Heart Disease](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6563505/)**")
        st.markdown("- **[Kidney Disease](https://indianjnephrol.org/prevalence-of-chronic-kidney-disease-among-adults-in-a-rural-community-in-south-india-results-from-the-kidney-disease-screening-kids-project/)**")
        st.markdown("- **[Liver Disease](https://www.pratikpatil.co.in/7-major-risk-factors-of-liver-cancer-in-india/)**")
        st.markdown("- **[Diabetes](https://www.thelancet.com/journals/langlo/article/PIIS2214-109X(18)30387-5/fulltext)**")
        st.markdown("- **[Parkinson's Disease](https://www.mdsabstracts.org/abstract/clinical-and-epidemiological-profile-of-parkinsons-disease-in-india/)**")

    with st.expander("⭕ Diagnosis & Screening"):
        st.markdown("- **[Brain Disease](https://www.apollohospitals.com/health-library/understanding-brain-tumors-types-symptoms-and-diagnosis/)**")
        st.markdown("- **[Heart Disease](https://www.mayoclinic.org/diseases-conditions/heart-disease/diagnosis-treatment/drc-20353124/)**")
        st.markdown("- **[Kidney Disease](https://www.hindujahospital.com/blog/chronic-kidney-disease-diagnosis-precautions-treatment/)**")
        st.markdown("- **[Liver Disease](https://www.manipalhospitals.com/blog/15-symptoms-of-serious-liver-disease/)**")
        st.markdown("- **[Diabetes](https://www.onetouchdiabetes.co.in/facts-about-diabetes/start-your-journey/signs-and-symptoms-of-diabetes)**")
        st.markdown("- **[Parkinson's Disease](https://pmc.ncbi.nlm.nih.gov/articles/PMC9764889/)**")

    with st.expander("⭕ Treatment Options"):
        st.markdown("- **[Brain Disease](https://www.maxhealthcare.in/blogs/decoding-brain-tumour-and-its-cure/)**")
        st.markdown("- **[Heart Disease](https://www.mayoclinic.org/diseases-conditions/heart-disease/diagnosis-treatment/drc-20353124/)**")
        st.markdown("- **[Kidney Disease](https://chaitanyastemcell.com/kidney-treatment-in-india/)**")
        st.markdown("- **[Liver Disease](https://www.pratikpatil.co.in/7-major-risk-factors-of-liver-cancer-in-india/)**")
        st.markdown("- **[Diabetes](https://www.onetouchdiabetes.co.in/facts-about-diabetes/start-your-journey/signs-and-symptoms-of-diabetes)**")
        st.markdown("- **[Parkinson's Disease](https://www.mayoclinic.org/diseases-conditions/parkinsons-disease/diagnosis-treatment/drc-20376062)**")

    with st.expander("⭕ Lifestyle & Diet Recommendations"):
        st.markdown("- **[Brain Disease](https://www.apollohospitals.com/health-library/understanding-brain-tumors-types-symptoms-and-diagnosis/)**")
        st.markdown("- **[Heart Disease](https://www.verywellhealth.com/strategies-prevent-heart-disease-4178066/)**")
        st.markdown("- **[Kidney Disease](https://chaitanyastemcell.com/kidney-treatment-in-india/)**")
        st.markdown("- **[Liver Disease](https://www.manipalhospitals.com/blog/15-symptoms-of-serious-liver-disease/)**")
        st.markdown("- **[Diabetes](https://www.onetouchdiabetes.co.in/facts-about-diabetes/start-your-journey/signs-and-symptoms-of-diabetes)**")
        st.markdown("- **[Parkinson's Disease](https://pmc.ncbi.nlm.nih.gov/articles/PMC9764889/)**")

    with st.expander("⭕ Interactive Tools & Tests"):
        st.markdown("- 🧠 **[Glasgow Coma Scale (Brain)](https://www.mdcalc.com/glasgow-coma-scale-score-gcs)**")
        st.markdown("- ❤️ **[Heart Risk Estimator](https://tools.acc.org/ascvd-risk-estimator-plus/)**")
        st.markdown("- 🧪 **[Kidney Disease: Clinician Tools](https://www.kidney.org/professionals/tools)**")
        st.markdown("- 🧪 **[Liver Disease: FibroScan for Early Screening](https://www.biospectrumindia.com/news/92/25162/city-imaging-launches-advanced-diagnostic-tool-fibroscan-for-early-screening-of-liver-diseases.html)**")
        st.markdown("- 🧠 **[Parkinson's Disease: MindMotion® GO Neuro Rehabilitation Tool](https://www.hindujahospital.com/blog/interactive-digital-neuro-rehabilitation-parkinsons-disease/)**")
        st.markdown("- 🩸 **[Diabetes Risk Calculator](https://www.onetouchdiabetes.co.in/facts-about-diabetes/start-your-journey/signs-and-symptoms-of-diabetes)**")
    
    with st.expander("⭕ Latest Research & News"):
        st.markdown("- **[Brain Disease](https://www.maxhealthcare.in/blogs/decoding-brain-tumour-and-its-cure/)**")
        st.markdown("- **[Heart Disease](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6563505/)**")
        st.markdown("- **[Kidney Disease](https://indianjnephrol.org/prevalence-of-chronic-kidney-disease-among-adults-in-a-rural-community-in-south-india-results-from-the-kidney-disease-screening-kids-project/)**")
        st.markdown("- **[Liver Disease](https://www.pratikpatil.co.in/7-major-risk-factors-of-liver-cancer-in-india/)**")
        st.markdown("- **[Diabetes](https://www.thelancet.com/journals/langlo/article/PIIS2214-109X(18)30387-5/fulltext)**")
        st.markdown("- **[Parkinson's Disease](https://pmc.ncbi.nlm.nih.gov/articles/PMC9764889/)**")

    with st.expander("⭕ Expert Interviews & Videos"):
        st.markdown("- 🎥 **[Brain Disease](https://www.apollohospitals.com/health-library/understanding-brain-tumors-types-symptoms-and-diagnosis/)**")
        st.markdown("- 🎥 **[Heart Disease](https://www.maxhealthcare.in/blogs/heart-disease-symptoms-and-causes/)**")
        st.markdown("- 🎥 **[Diabetes: India's Leading Diabetes Doctor - Dr. Roshani Sanghani](https://www.youtube.com/watch?v=HL3XBBh_s1Y)**")
        st.markdown("- 🎥 **[Kidney Disease: Dr. Ravi V Andrews - Apollo Hospitals](https://www.youtube.com/watch?v=8PXdRaNiPzU)**")
        st.markdown("- 🎥 **[Liver Disease: Dr. Anurag Shetty on Liver Cirrhosis - Manipal Hospitals](https://www.youtube.com/watch?v=omY6BlgiEBs)**")
        st.markdown("- 🎥 **[Parkinson's Disease: Patient Testimonial - Manipal Hospitals](https://www.youtube.com/watch?v=0u65oF1Sr5Y)**")
