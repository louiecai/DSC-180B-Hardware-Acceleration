<h1> Enhancing Human Activity Recognition with Hardware Acceleration </h1>

<!-- Link to this website: https://louiecai.github.io/DSC-180B-Hardware-Acceleration/ -->

## Home

Welcome to our project page! This research focuses on improving Human Activity Recognition (HAR) through innovative hardware acceleration techniques.

## **About**

This project pioneers the enhancement of Human Activity Recognition (HAR) through <b>specialized hardware acceleration</b>, moving beyond traditional software optimization approaches. By addressing the inefficiencies of current technologies in processing HAR tasks, our aim is to <b>dramatically improve system performance, reduce response times, and eliminate operational bottlenecks</b>. This breakthrough is set to enhance the effectiveness and efficiency of HAR applications, making a significant impact across various fields such as wearable technology, healthcare, and intelligent home systems.

The significance of our endeavor is underscored by the critical role that HAR systems play in modern applications—from monitoring health vitals to facilitating seamless interaction within smart environments. By focusing on hardware acceleration, we target the core challenges of slow processing and high energy demands prevalent in existing systems. The outcome is anticipated to be more agile, reliable, and energy-efficient solutions that cater more effectively to their intended uses, thereby <b>improving the quality of life and operational efficiency</b> in numerous applications.

The potential impact of our project is profound. In the realm of wearable technology, it promises to enhance device performance and user experience. In healthcare, it could lead to more accurate and real-time monitoring of patients, potentially saving lives.
For smart home systems, our advancements aim to bring about smarter, more responsive environments that adapt to the needs of their inhabitants.
Through this project, we are not just advancing technology but also paving the way for a future where technology more seamlessly integrates into and enhances every aspect of our daily lives.

## Methodology

### Data Collection and Analysis Process

The data collection process for our project is meticulously designed to evaluate the performance of Human Activity Recognition (HAR) neural networks across a spectrum of hardware configurations. Key components of this process include:

- **AWS Setup**: We leverage AWS instances to provide the computational power needed for our experiments. Each instance is tailored to match the requirements of different HAR neural network configurations.
- **Docker Containers**: To ensure consistency and reproducibility, Docker containers are deployed. These containers are pre-configured with all necessary dependencies to facilitate the running of our models across various environments.
- **Execution Scripts**: Automation scripts are crucial for streamlining the data collection process. These scripts orchestrate the setup, execution, and data gathering across different hardware setups, ensuring efficiency and accuracy in our findings.

<details>
<summary><b>More on Hardware Configurations and Data Collection</b></summary>

<p>In the context of our research, we explore a diverse array of hardware configurations to understand their impact on HAR system performance. This exploration includes but is not limited to:</p>
<ul>
    <li>Low-end CPUs that mirror the processing capabilities of wearable tech devices.</li>
    <li>Multi-core CPUs to assess the benefits of parallel processing.</li>
    <li>GPUs with varying tensor core counts to evaluate acceleration benefits for neural network computations.</li>
</ul>
<p>The data collection methodology is thorough, involving steps to ensure that each model is evaluated under identical conditions across the hardware spectrum. Post-collection, data is processed and analyzed to draw insights into performance variations, efficiency gains, and potential bottlenecks in HAR applications. This comprehensive approach allows us to pinpoint optimal hardware configurations that balance computational power, energy consumption, and real-world applicability.</p>

<p>Here are some examples of the hardware configurations we explored:</p>

<table>
    <thead>
        <tr>
            <th>Environment</th>
            <th>Instance</th>
            <th>vCPU</th>
            <th>Memory (GiB)</th>
            <th>CPU Type</th>
            <th>GPU</th>
            <th>GPU Memory (GiB)</th>
            <th>GPU Type</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>env1</td>
            <td>c7a.medium</td>
            <td>1</td>
            <td>2.0</td>
            <td>AMD EPYC Gen4</td>
            <td>0</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>env2</td>
            <td>c7a.large</td>
            <td>2</td>
            <td>4.0</td>
            <td>AMD EPYC Gen4</td>
            <td>0</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>env3</td>
            <td>c7a.xlarge</td>
            <td>4</td>
            <td>8.0</td>
            <td>AMD EPYC Gen4</td>
            <td>0</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>...</td>
            <td>...</td>
            <td>...</td>
            <td>...</td>
            <td>...</td>
            <td>...</td>
            <td>...</td>
            <td>...</td>
        </tr>
        <tr>
            <td>env16</td>
            <td>g5.4xlarge</td>
            <td>16</td>
            <td>64.0</td>
            <td>AMD EPYC Gen2</td>
            <td>1</td>
            <td>24.0</td>
            <td>NVIDIA A10G</td>
        </tr>
        <tr>
            <td>datahub1</td>
            <td>-</td>
            <td>1</td>
            <td>32.0</td>
            <td>Intel(R) Xeon(R) Gold 5218</td>
            <td>1</td>
            <td>-</td>
            <td>2080ti</td>
        </tr>
        <tr>
            <td>datahub10</td>
            <td>-</td>
            <td>12</td>
            <td>32.0</td>
            <td>AMD EPYC 7543P</td>
            <td>1</td>
            <td>-</td>
            <td>a5000</td>
        </tr>
    </tbody>
</table>

<p>Note: "-" indicates that the system does not include a GPU or the specific information is not applicable.</p>

</details>

## Results

<div style="display: grid; place-items: center; min-height: 100vh;">

<iframe src="runtime_cpu.html" width="90%" frameborder="0" height="500"></iframe>

<iframe src="runtime_vcpu.html" width="90%" frameborder="0" height="500"></iframe>

</div>

## Discussion

(Placeholder for discussion on the findings, implications, and potential future work)

## References

(Placeholder for citing all references and resources used in the project)

## Contact

Information on how to get in touch with the team for questions or collaborations.
