import { useState, useEffect, useRef } from "react";
import {
  Box, Select, Input, Button, Text, VStack,
  Spinner, Progress, Menu, MenuButton, MenuList, MenuItem, Tooltip
} from "@chakra-ui/react";
import { ChevronDownIcon } from "@chakra-ui/icons";
import { useDropzone } from "react-dropzone";

const API_URL = "http://localhost:8000";

function App() {
  const [inputMode, setInputMode] = useState("url");
  const [selectedFile, setSelectedFile] = useState(null);
  const [videoUrl, setVideoUrl] = useState("");
  const [uploadedFilename, setUploadedFilename] = useState(null);
  const [isSegmented, setIsSegmented] = useState(false);
  const [streamUrl, setStreamUrl] = useState(null);
  const [status, setStatus] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [violations, setViolations] = useState([]);
  const [videoStartTime, setVideoStartTime] = useState(null);
  const [videoSource, setVideoSource] = useState(null);
  const [pollingActive, setPollingActive] = useState(false);

  const intervalRef = useRef(null);

  const resetState = () => {
    setSelectedFile(null);
    setUploadedFilename(null);
    setStreamUrl(null);
    setStatus("");
    setIsUploading(false);
    setIsProcessing(false);
    setViolations([]);
    setVideoStartTime(null);
    setVideoSource(null);
    setPollingActive(false);
    setIsSegmented(false);
  };

  const handleInputModeChange = (e) => {
    const mode = e.target.value;
    setInputMode(mode);
    resetState();
  };

  const handleUrlChange = (e) => {
    setVideoUrl(e.target.value);
  };

  const handleReload = () => {
    if (!isSegmented || !videoUrl.trim()) return;
    resetState();
    const now = Date.now();
    const trimmed = videoUrl.trim();
    setStreamUrl(`${API_URL}/stream?source=${encodeURIComponent(trimmed)}&reload=${now}&refresh=true`);
    setVideoStartTime(new Date());
    setVideoSource(trimmed);
    setViolations([]);
    setPollingActive(false);
  };
  

  const handleStopStream = () => {
    handleStop();        // ⬅️ stop dulu backend
    resetState();        // ⬅️ baru reset frontend state
  };

  const handleStop = async () => {
    if (!videoSource) {
      alert("Tidak ada video yang sedang diproses.");
      return;
    }

    try {
      console.log("Stopping source:", videoSource); // debug

      await fetch(`${API_URL}/stop-stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: videoSource }) // GUNAKAN state TERAKHIR
      });

      const video = document.getElementById("video-stream");
      if (video) {
        video.src = "";
      }

      // alert("Streaming dihentikan.");
    } catch (err) {
      console.error("Gagal menghentikan stream:", err);
    }
  };

  
  useEffect(() => {
    if (!isProcessing && videoStartTime && videoSource) setPollingActive(true);
  }, [isProcessing, videoStartTime, videoSource]);

  const handleSubmit = (fileOverride = null) => {
    if (inputMode === "url") {
      if (!videoUrl.trim()) return alert("Masukkan URL video dulu!");
      resetState();
      const trimmed = videoUrl.trim();
      const now = Date.now();
      setStreamUrl(`${API_URL}/stream?source=${encodeURIComponent(trimmed)}&reload=${now}`);
      setVideoStartTime(new Date());
      setVideoSource(trimmed);
      setIsProcessing(true);


    } else if (inputMode === "upload") {
      const file = fileOverride || selectedFile;
      if (!file) return alert("Pilih file video dulu!");
      resetState();
      setIsUploading(true);

      const formData = new FormData();
      formData.append("file", file);

      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${API_URL}/upload`);

      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
          setUploadProgress(Math.round((e.loaded / e.total) * 100));
        }
      };

      xhr.onload = () => {
        setIsUploading(false);
        if (xhr.status === 200) {
          const data = JSON.parse(xhr.responseText);
          const stream = `${API_URL}/stream?source=${encodeURIComponent(data.filename)}`;
          setUploadedFilename(data.filename);
          setStreamUrl(stream);
          setVideoStartTime(new Date());
          setVideoSource(data.filename);
          setIsProcessing(true);
        } else setStatus("Upload gagal!");
      };

      xhr.onerror = () => {
        setStatus("Upload error!");
        setIsUploading(false);
      };

      xhr.send(formData);
    }
  };

  const onDrop = (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file?.type.startsWith("video")) {
      setSelectedFile(file);
      handleSubmit(file);
    } else alert("Hanya file video yang diperbolehkan!");
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "video/*": [] },
    disabled: inputMode === "url",
  });

  const renderViolationItem = (v, i) => (
      <li
        key={i}
        onClick={() => v && window.open(`http://localhost:8000${v.image_url}`, "_blank")}
        style={{
          display: "flex",
          alignItems: "center",
          marginBottom: "16px",
          border: "1px solid #333",
          background: "linear-gradient(135deg, #2c2c2c, #1e1e1e)",
          padding: "10px",
          borderRadius: "12px",
          cursor: v ? "pointer" : "default",
          transition: "transform 0.2s ease-in-out",
          transform: "scale(1.0)",  // ⬅ ini penting
          boxShadow: "0 1px 1px rgba(0, 0, 0, 0.65)"
        }}
        onMouseEnter={(e) => v && (e.currentTarget.style.transform = "scale(1.07)")}
        onMouseLeave={(e) => v && (e.currentTarget.style.transform = "scale(1.0)")}
      >


      <img
        src={v ? `${API_URL}${v.image_url}` : `/icons8-car-90.png`}
        alt={`Pelanggaran ${i + 1}`}
        style={{
          width: 100, height: 100, opacity: 0.85, borderRadius: 8,
          objectFit: "contain", marginRight: 12, marginLeft: 6, border: "1px solid #555"
        }}
      />
      <div style={{ flex: 1 }}>
        <strong style={{ fontSize: 16, color: "#59caf8", marginLeft: 6}}>
          Pelanggaran #{i + 1}
        </strong>
        <p style={{ fontSize: 14, color: "#ccc", margin: "4px 0 2px", marginLeft: 8 }}>
          ID - {v ? v.track_id : "None"}
        </p>
        <p style={{ fontSize: 14, color: "#ccc", margin: "4px 0 2px", marginLeft: 6 }}>
          🕒 {v ? v.timestamp : "- None"}
        </p>
      </div>
    </li>
  );

  return (
    <div style={{ backgroundColor: "#111111", minHeight: "100vh", color: "#f0f0f0" }}>
      {/* Header */}
      <div style={{
        backgroundColor: "#1f1f1f",
        padding: "18px 45px 16px 45px",
        display: "flex",
        boxShadow: "0 5px 7px rgba(0,0,0,0.5)"
      }}>
        <nav style={{ display: "flex", alignItems: "center", gap: "30px" }}>
          <Button as="a" href="/" leftIcon={<img src="/home-6.png" alt="icon" width={25} height={25} style={{ marginRight: 5, opacity: 0.85 }} />}
            variant="outline" rounded="md" height="45px"
            borderWidth="0px" color="#a2cefe"
            _hover={{ bg: "#14204a" }} >
            Home
          </Button>
          <Button as="a" href="/dashboard" leftIcon={<img src="/dashboard-3.png" alt="icon" width={25} height={25} style={{ marginRight: 5, opacity: 0.9 }} />}
            variant="outline" rounded="md" height="45px"
            borderWidth="0px" borderColor="#646466"
            _hover={{ bg: "#27272a" }} color="white">
            Dashboard
          </Button>
        </nav>
      </div>

      {/* Content */}
      <div style={{ display: "flex", padding: 10, marginLeft: 40 }}>
        <div style={{ flex: 3, marginRight: 20 }}>
          <h1 style={{ fontSize: 30, fontWeight: 'bold', marginBottom: 15, marginTop: 20 }}>
            Wrong-Way Vehicle Violation Detection
          </h1>

          <div style={{ display: "flex", alignItems: "center", marginBottom: 20 }}>
            <Menu>
              <MenuButton as={Button} rightIcon={<ChevronDownIcon />} minWidth="160px" height="45px"
                fontSize="md" fontWeight="semibold" bg="#111" color="white"
                border="1px solid #333" _hover={{ bg: "#151515" }} _expanded={{ bg: "#151515" }}>
                {inputMode === "url" ? "Input URL" : "Upload File"}
              </MenuButton>
              <MenuList paddingLeft={4} bg="#1f1f1f" borderColor="#333" py="0.5" borderRadius="md" boxShadow="md" minWidth="160px" zIndex={9999}>
                <MenuItem bg="#1f1f1f" _hover={{ bg: "#252525" }} color="white" fontSize="15px" mt="1"
                  onClick={() => handleInputModeChange({ target: { value: "url" } })}>
                  Input URL
                </MenuItem>
                <MenuItem bg="#1f1f1f" _hover={{ bg: "#252525" }} color="white" fontSize="15px"
                  onClick={() => handleInputModeChange({ target: { value: "upload" } })}>
                  Upload File
                </MenuItem>

              </MenuList>
            </Menu>
            {inputMode === "url" && (
            <>
              <Input
                type="text"
                height="45px"
                placeholder="Masukkan URL video..."
                value={videoUrl}
                onChange={handleUrlChange}
                borderColor="#333"
                ml={3}
                _hover={{ bg: "#151515" }}
                focusBorderColor="#aaa"
                color="white"
                _placeholder={{ color: "#777" }}
              />
              <Button
                onClick={() => handleSubmit()}
                ml={3}
                bg="#1689c7"
                color="white"
                fontSize="md"
                fontWeight="semibold"
                minWidth="80px"
                _hover={{ bg: "#167ac7" }} // opsional hover effect
              >
                Submit
              </Button>
            </>
          )}
          </div>

          {status && <Text mb={2}>{status}</Text>}
          {isUploading && (
            <Box mb={4}>
              <Text fontSize="md" fontWeight="medium" color="white" mb={3}>
                {selectedFile?.name || "Uploading..."}
              </Text>
              <Progress
                value={uploadProgress}
                size="sm"
                borderRadius="md"
                bg="gray.200" // warna track belakang
                sx={{
                  '& > div': {
                    background: '#3ebcee' // warna bar isi
                  }
              }} />
              <Text mt={1} fontSize="sm" color="gray.300">
                Uploading... {uploadProgress}%
              </Text>
            </Box>
          )}

          <Box
            position="relative"
            width="85%"
            mx="auto"
            aspectRatio="16 / 9"
            border="1.5px dashed #9999"
            color="#888"
            borderRadius="xl"
            transition="all 0.3s"
            mb="6"
            overflow="hidden"
            zIndex={0}
            {...(inputMode === "upload" ? getRootProps() : {})}
            _hover={{
              backgroundColor: inputMode === "upload" ? "#151515" : "transparent",
              borderColor: "#888",
            }}
          >
            {isProcessing && streamUrl && (
              <Box
                position="absolute"
                top={0}
                left={0}
                right={0}
                bottom={0}
                display="flex"
                justifyContent="center"
                alignItems="center"
                flexDirection="column"
                bg="rgba(0,0,0,0.5)"
                zIndex={1}
              >
                <Spinner thickness="5px" speed="0.65s" emptyColor="gray.200" color="blue.400" size="xl" />
                <Text mt={4} fontWeight="semibold" fontSize="md" color="white">
                  Processing road segmentation
                </Text>
              </Box>
            )}

            <Box
              position="absolute"
              top={0}
              left={0}
              right={0}
              bottom={0}
              display="flex"
              justifyContent="center"
              alignItems="center"
              flexDirection="column"
            >
              {inputMode === "upload" && <input {...getInputProps()} />}
              {streamUrl ? (
                <>
                  <img
                    src={streamUrl}
                    alt="Live Stream"
                    style={{
                      width: "100%",
                      height: "100%",
                      objectFit: "contain",
                      backgroundColor: "black",
                    }}
                    onLoad={() => {
                      if (isProcessing) {
                        setIsProcessing(false);
                        setIsSegmented(true);
                      }
                    }}
                  />

                  {/* Tombol X pojok kanan atas */}
                  <Tooltip label="Stop Streaming" hasArrow placement="top" bg="gray.700" color="white" fontSize="md">
                    <Box
                      position="absolute"
                      top="10px"
                      right="10px"
                      zIndex="3"
                      onClick={handleStopStream}
                      cursor="pointer"
                      display="flex"
                      alignItems="center"
                      justifyContent="center"
                      // bg="rgba(0, 0, 0, 0.5)"
                      padding="6px"
                      borderRadius="md"
                      // _hover={{ bg: "rgba(0, 0, 0, 0.75)" }}
                    >
                    <img
                      src="/exit.png"
                      alt="Stop"
                      onClick={handleStop}
                      style={{
                        width: "80%",               // ⬅️ ukuran dinamis berdasarkan lebar container
                        height: "auto",             // ⬅️ biar proporsional
                        opacity: 0.8,
                        filter: "brightness(0.9)",
                      }}
                    />
                    </Box>
                  </Tooltip>

                </>
              ) : inputMode === "upload" ? (
                <VStack spacing={1}>
                  <img src="/icons8-upload-96.png" alt="Upload Icon" width={35} height={35} style={{ opacity: 0.6 }} />
                  {isDragActive ? (
                    <Text>Place your video here...</Text>
                  ) : (
                    <>
                      <Text>Drag and drop your video here to start streaming</Text>
                      <Text>.mp4</Text>
                    </>
                  )}
                </VStack>
              ) : (
                <VStack spacing={1}>
                  <img src="/security-camera-3.png" alt="Video Icon" width={50} height={50} style={{ opacity: 0.5 }} />
                  <Text>Input URL address</Text>
                </VStack>
              )}
            </Box>
          </Box>
        </div>

        {/* Sidebar pelanggaran */}
        {/* <div style={{
          flex: 1,
          backgroundColor: "#1f1f1f",
          boxShadow: "6px 6px 6px 8px rgba(0, 0, 0, 0.4)",
          padding: "25px 20px 10px",
          borderRadius: "16px",
          color: "#fff",
          height: "100%",
          overflowY: "auto"
        }}>
          <h3 style={{
            fontSize: "21px",
            textAlign: "center",
            fontWeight: "bold",
            marginBottom: "20px",
            borderBottom: "1px solid #333",
            paddingBottom: "10px",
            letterSpacing: "0.5px"
          }}>
            Pelanggaran yang Terdeteksi
          </h3>
          <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
            {Array.from({ length: 5 }).map((_, i) => renderViolationItem(violations[i], i))}
          </ul>

        </div> */}
      </div>
    </div>
  );
}

export default App;
