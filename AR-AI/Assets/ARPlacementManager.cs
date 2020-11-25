using System;
using System.IO;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;
using UnityEngine.UI;
using TensorFlowLite;

[RequireComponent(typeof(ARRaycastManager))]
public class ARPlacementManager : MonoBehaviour
{
    [SerializeField] 
    string classificationFile = "mobilenet_v2_1.0_224.tflite";
    
    [SerializeField] 
    string detectionFile = "ssd_mobilenet_v1_1_metadata_1.tflite";

    [SerializeField]
    bool doObjectDetection = false;

    [SerializeField]
    private GameObject arCamera;

    [SerializeField]
    private ARCameraBackground arCameraBackground;

    [SerializeField]
    int interval = 1;


    private ARRaycastManager raycastManager;
    private static List<ARRaycastHit> hits = new List<ARRaycastHit>();

    RenderTexture cameraTexture;
    RawImage cameraView = null;

    Interpreter classifier;
    private const int classifierInputSize = 224;
    private const int classifierOutputSize = 1001;
    private float[,,] classifierInputs = new float[classifierInputSize, classifierInputSize, 3];
    private float[] classifierOutputs = new float[classifierOutputSize];

    SSD detector;

    TextureToTensor textureToTensor;

    DateTime last;

    private void Awake()
    {
        raycastManager = GetComponent<ARRaycastManager>();
    }

    void Start()
    {
        var options = new InterpreterOptions()
        {
            threads = 4,
            useNNAPI = false,
        };

        if (doObjectDetection)
        {

            string path = Path.Combine(Application.streamingAssetsPath, detectionFile);
            detector = new SSD(path);

            textureToTensor = new TextureToTensor();

            cameraTexture = new RenderTexture(Screen.width, Screen.height, 0);
            cameraView.texture = cameraTexture;
        }
        else
        {
            classifier = new Interpreter(FileUtil.LoadFile(classificationFile), options);
            Debug.Log($"initialized classifier");
            Debug.Log($"classifier: input: {classifier.GetInputTensorInfo(0)}");
            Debug.Log($"classifier: input: {classifier.GetInputTensorInfo(0).shape[1]}");
            Debug.Log($"classifier: output: {classifier.GetOutputTensorInfo(0)}");
            Debug.Log($"classifier: output: {classifier.GetOutputTensorInfo(0).shape[1]}");

            classifier.ResizeInputTensor(0, new int[] { 1, classifierInputSize, classifierInputSize, 3 });
            classifier.AllocateTensors();

            textureToTensor = new TextureToTensor();

            cameraTexture = new RenderTexture(classifierInputSize, classifierInputSize, 0);            
        }

        last = DateTime.Now;
    }

    void Update()
    {
        if (arCameraBackground.material != null)
        {
            DateTime now = DateTime.Now;
            if ((now-last).TotalSeconds >= interval)
            {
                if (doObjectDetection)
                {
                    Detect();
                }
                else
                {
                    Classify();
                }
                last = now;
            }
        }
    }


    private void Detect()
    {
        Graphics.Blit(null, cameraTexture, arCameraBackground.material);

        detector.Invoke(cameraTexture);
        var results = detector.GetResults();

        var catPos = -1;
        var catProb = 0f;
        for (int i = 0; i < results.Length; i++)
        {
            if (results[i].classID == 16)
            {
                if (results[i].score > catProb)
                {
                    catPos = i;
                    catProb = results[i].score;
                }
            }
        }

        if (catPos != -1)
        {
            var x = (results[catPos].rect.x + (results[catPos].rect.width / 2)) * Screen.width;
            var y = (results[catPos].rect.y + (results[catPos].rect.height / 2)) * Screen.height;
            Debug.Log($"X Y: {x}, {y}");
            var pos = GetPosition(x, y);
            Debug.Log($"get position: {pos.x}, {pos.y}, {pos.z}");

            GameObject tmp = GameObject.CreatePrimitive(PrimitiveType.Cube);
            tmp.transform.localScale = new Vector3(0.1f, 0.1f, 0.1f);
            Debug.Log("Placed object");
            Instantiate(tmp, pos, Quaternion.identity);
        }
    }

    private void Classify()
    {
        Graphics.Blit(null, cameraTexture, arCameraBackground.material);

        textureToTensor.ToTensor(cameraTexture, classifierInputs);

        classifier.SetInputTensorData(0, classifierInputs);
        classifier.Invoke();
        classifier.GetOutputTensorData(0, classifierOutputs);

        var (maxPos, maxProb) = Softmax(classifierOutputs);
        Debug.Log($"predicted: {maxPos} {maxProb}");

        if (maxPos >= 283 && maxPos <= 295)
        {
            var pos = GetPosition(Screen.width / 2, Screen.height / 2);
            Debug.Log($"get position: {pos.x}, {pos.y}, {pos.z}");

            GameObject tmp = GameObject.CreatePrimitive(PrimitiveType.Cube);
            tmp.transform.localScale = new Vector3(0.1f, 0.1f, 0.1f);
            Debug.Log("Placed object");
            Instantiate(tmp, pos, Quaternion.identity);
        }
    }


    private (int, float) Softmax(float[] predictions)
    {
        var maxProb = 0f;
        var maxPos = 0;
        for (int i=0;i<predictions.Length;i++)
        {
            if (predictions[i] > maxProb)
            {
                maxProb = predictions[i];
                maxPos = i;
            }
        }
        return (maxPos, maxProb);
    }

    private Vector3 GetPosition(float x, float y)
    {
        var hits = new List<ARRaycastHit>();
        raycastManager.Raycast(new Vector2(x, y), hits, TrackableType.All);

        if (hits.Count > 0)
        {
            var pose = hits[0].pose;
            return pose.position;
        }
        return new Vector3();
    }

}