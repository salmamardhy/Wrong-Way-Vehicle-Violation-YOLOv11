import { useEffect, useState } from "react";
import {
  Box, Text, SimpleGrid, Image, Select, Button, HStack,
  Menu, MenuButton, MenuList, MenuItem, Flex
} from "@chakra-ui/react";
import { ChevronDownIcon } from "@chakra-ui/icons";
import { useSearchParams, Link } from "react-router-dom";
import { LuChevronLeft, LuChevronRight } from "react-icons/lu";

function splitURLToJSX(url) {
  const parts = url.split('/');
  if (parts.length < 4) return url;
  const domain = parts.slice(0, 3).join('/') + '/';
  const path = parts.slice(3).join('/');
  return (
    <>{domain}<br />{path}</>
  );
}

function Dashboard() {
  const [violations, setViolations] = useState([]);
  const [sortBy, setSortBy] = useState("timestamp");
  const [order, setOrder] = useState("desc");
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const limit = 20;

  useEffect(() => {
    const fetchViolations = async () => {
      const res = await fetch(`http://localhost:8000/violations?page=${page}&sort_by=${sortBy}&order=${order}&_=${Date.now()}`);
      const data = await res.json();
      setViolations(data);
    };

    const fetchTotal = async () => {
      try {
        const res = await fetch("http://localhost:8000/violations/count");
        const data = await res.json();
        setTotal(data.total || 0);
      } catch (err) {
        console.error("Failed to fetch count:", err);
        setTotal(0);
      }
    };

    fetchViolations();
    fetchTotal();
  }, [sortBy, order, page]);

  useEffect(() => {
    // Ganti default order saat sortBy berubah
    if (sortBy === "timestamp") {
      setOrder("desc");
    } else {
      setOrder("asc");
    }
  }, [sortBy]);

  const totalPages = Math.ceil(total / limit);

  return (
    <Box bg="#111111" minH="100vh" color="#f0f0f0">
      {/* Header */}
      <Box bg="#1f1f1f" p="18px 45px 16px 45px" display="flex" boxShadow="0 5px 7px rgba(0,0,0,0.5)">
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
      </Box>

      {/* Content */}
      <Box style={{ padding: 10, paddingBottom:30 , marginLeft: 40, marginTop: 20, marginRight: 30}}>
        <h1 style={{fontSize: 30, fontWeight: 'bold', marginBottom:"15px"}}>
          Traffic Violation Dashboard
        </h1> 

        {/* Filter Sort */}
        <HStack spacing={4} mb={6}>
          {/* Sort By */}
          <Menu>
            <MenuButton as={Button} rightIcon={<ChevronDownIcon />} minWidth="160px" height="45px"
              fontSize="md" fontWeight="semibold" bg="#111" color="white"
              border="1px solid #333" _hover={{ bg: "#151515" }} _expanded={{ bg: "#151515" }}
              justifyContent="flex-start" textAlign="left" px={7}>
              {sortBy === "timestamp" ? "Timestamp" : sortBy === "track_id" ? "ID" : "Source"}
            </MenuButton>
            <MenuList paddingLeft={3} bg="#1f1f1f" borderColor="#333" borderRadius="md" boxShadow="md" minWidth="160px">
              <MenuItem bg="#1f1f1f" _hover={{ bg: "#252525" }} color="white" fontSize="sm"
                onClick={() => setSortBy("timestamp")}>Timestamp</MenuItem>
              <MenuItem bg="#1f1f1f" _hover={{ bg: "#252525" }} color="white" fontSize="sm"
                onClick={() => setSortBy("track_id")}>ID</MenuItem>
              <MenuItem bg="#1f1f1f" _hover={{ bg: "#252525" }} color="white" fontSize="sm"
                onClick={() => setSortBy("source")}>Source</MenuItem>
            </MenuList>
          </Menu>

          {/* Order */}
          <Menu>
            <MenuButton as={Button} rightIcon={<ChevronDownIcon />} minWidth="120px" height="45px"
              fontSize="md" fontWeight="semibold" bg="#111" color="white"
              border="1px solid #333" _hover={{ bg: "#151515" }} _expanded={{ bg: "#151515" }}
              justifyContent="flex-start" textAlign="left" px={7}>
              {sortBy === "timestamp"
                ? (order === "desc" ? "Newest" : "Oldest")
                : (order === "asc" ? "A-Z" : "Z-A")}
            </MenuButton>
            <MenuList px={3} bg="#1f1f1f" borderColor="#333" borderRadius="md" boxShadow="md" minWidth="135px">
              {sortBy === "timestamp" ? (
                <>
                  <MenuItem bg="#1f1f1f" _hover={{ bg: "#252525" }} color="white" fontSize="sm"
                    onClick={() => setOrder("desc")}>Newest</MenuItem>
                  <MenuItem bg="#1f1f1f" _hover={{ bg: "#252525" }} color="white" fontSize="sm"
                    onClick={() => setOrder("asc")}>Oldest</MenuItem>
                </>
              ) : (
                <>
                  <MenuItem bg="#1f1f1f" _hover={{ bg: "#252525" }} color="white" fontSize="sm"
                    onClick={() => setOrder("asc")}>A-Z</MenuItem>
                  <MenuItem bg="#1f1f1f" _hover={{ bg: "#252525" }} color="white" fontSize="sm"
                    onClick={() => setOrder("desc")}>Z-A</MenuItem>
                </>
              )}
            </MenuList>
          </Menu>
        </HStack>

        {/* List Pelanggaran */}
        <SimpleGrid columns={[1, 2, 3]} spacing={6}>
          {violations.map((v) => (
            <Box
              key={v.id}
              bg="#1f1f1f"
              p="10px 15px 20px 25px"
              rounded="xl"
              border= "1px solid #2e2e2e"
              boxShadow= "0 3px 8px rgba(0, 0, 0, 0.65)"
              bgGradient="linear(120deg,rgb(38, 38, 38),rgb(24, 24, 24))"
              _hover={{ transform: "scale(1.05)", borderColor: "#3182CE", borderWidth: "2px"}}
              transition="transform 0.2s ease-in-out, border-color 0.2s ease-in-out"
            >
              {/* ID */}
              <Text fontWeight="bold" fontSize="17px" mb={2} mt={1} color="#59caf8">
                ID - <Link to={`/violation/${v.track_id}`}>{v.track_id}</Link>
              </Text>

              {/* Gambar + Keterangan */}
              <HStack spacing={4} align="flex-start">
              <Image
                src={`http://localhost:8000${v.image_url}`}
                alt={`Pelanggaran ${v.id}`}
                fallbackSrc="/notfound.jpg"
                boxSize="110px"
                objectFit="contain"
                backgroundColor="#000"
                borderRadius="md"
                flexShrink={0} // ⬅️ Tambahan penting agar tidak menyusut di Flex layout
              />

              <Box>
                <Flex mb={1}>
                  <Text fontSize="15px" fontWeight="semibold" color="gray.200" minWidth="60px">
                    Time
                  </Text>
                  <Text fontSize="15px" color="gray.300" mr={2}>:</Text>
                  <Text fontSize="15px" color="gray.300">
                    {v.timestamp}
                  </Text>
                </Flex>
                <Flex>
                  <Text fontSize="15px" fontWeight="semibold" color="gray.200" minWidth="60px">
                    Source
                  </Text>
                  <Text fontSize="15px" color="gray.300" mr={2}>:</Text>
                  <Text fontSize="15px" color="gray.300" whiteSpace="normal" wordBreak="break-word">
                    {splitURLToJSX(v.source)}
                  </Text>
                </Flex>
              </Box>

              </HStack>
            </Box>
          ))}
        </SimpleGrid>

        {/* Pagination */}
      <HStack mt={8} spacing={1} justify="center">
        <Button
          onClick={() => setPage(p => p - 1)}
          isDisabled={page === 1}
          leftIcon={<LuChevronLeft />}
          variant="ghost"
          colorScheme="white"
        />

        {Array.from({ length: totalPages }, (_, i) => i + 1)
          .filter(p =>
            p === 1 || p === totalPages || (p >= page - 2 && p <= page + 2)
          )
          .map((p, i, arr) => (
            <>
              {/* Tambahkan ellipsis jika ada gap */}
              {i > 0 && p - arr[i - 1] > 1 && (
                <Box key={`ellipsis-${i}`} px={2} color="#777">...</Box>
              )}
              <Button
                key={p}
                onClick={() => setPage(p)}
                variant={page === p ? "solid" : "ghost"}
                color={page === p ? "white" : "#636363"}
                bg={page === p ? "blue.500" : "transparent"}
                size="sm"
                rounded="md"
                px={3}
                _hover={{
                  bg: page === p ? "#263b85" : "gray.200",
                }}
              >
                {p}
              </Button>
            </>
          ))}

        <Button
          onClick={() => setPage(p => p + 1)}
          isDisabled={page === totalPages}
          rightIcon={<LuChevronRight />}
          variant="ghost"
          colorScheme="white"
        />
      </HStack>

      </Box>
    </Box>
  );
}

export default Dashboard;
