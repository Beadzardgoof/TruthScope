import { Backdrop, Button, CircularProgress, styled } from "@mui/material";
import { CloudUpload } from "@mui/icons-material";
import "./hero.css";
import * as React from "react";
import Dialog from "@mui/material/Dialog";
import DialogActions from "@mui/material/DialogActions";
import DialogContent from "@mui/material/DialogContent";
import DialogContentText from "@mui/material/DialogContentText";
import DialogTitle from "@mui/material/DialogTitle";
import Slide from "@mui/material/Slide";

const VisuallyHiddenInput = styled("input")({
  clip: "rect(0 0 0 0)",
  clipPath: "inset(50%)",
  height: 1,
  overflow: "hidden",
  position: "absolute",
  bottom: 0,
  left: 0,
  whiteSpace: "nowrap",
  width: 1,
});

function Hero() {
  const [isLoading, setIsLoading] = React.useState(false);
  const [weightedAverage, setweightedAverage] = React.useState();
  const [percAudio, setpercAudio] = React.useState();
  const [percText, setpercText] = React.useState();
  const [percExpression, setpercExpression] = React.useState();
  const [percVideo, setpercVideo] = React.useState();
  const handleOpen = () => {
    setIsLoading(true);
  };
  const handleClose = () => {
    setIsLoading(false);
  };
  const Transition = React.forwardRef(function Transition(props, ref) {
    return <Slide direction="up" ref={ref} {...props} />;
  });

  const [open, setOpen] = React.useState(false);
  const [detailsOpen, setDetailsOpen] = React.useState(false);

  const toggleDialog = () => {
    setOpen(!open);
  };
  const toggleDetailsDialog = () => {
    setDetailsOpen(!detailsOpen);
  };
  const resultStyle = {
    display: "flex",
    fontSize: "20px",
    height: "100px",
    fontWeight: "bold",
    color: "black",
    textAlign: "center",
    padding: "20px",
    margin: "auto",
    width: "85%",
    borderRadius: "30px",
  };
  const handleFileChange = (event) => {
    const fileList = event.target.files;
    if (fileList && fileList.length > 0) {
      const data = new FormData();
      data.append("file", fileList[0]);
      handleOpen();
      fetch("http://127.0.0.1:8000/api/upload/", {
        method: "POST",
        body: data,
      })
        .then((response) => response.json())
        .then((APIResult) => {
          console.log("Uploaded successfully", APIResult);
          setweightedAverage(APIResult.result.weighted_average);
          setpercAudio(APIResult.result.percentage_audio);
          setpercText(APIResult.result.percentage_text);
          setpercVideo(APIResult.result.percentage_video);
          setpercExpression(APIResult.result.percentage_expressions);
          toggleDialog();
          handleClose();
        })
        .catch((error) => {
          console.error("Error:", error);
          handleClose();
        });
    }
    console.log("Selected file(s):", fileList);
  };
  return (
    <div className="hero">
      <div className="hero-left">
        <div className="hero-header">
          Unveiling Truth: Your Honesty Guardian
        </div>
        <div className="hero-desc">
          Our cutting-edge lie detection technology empowers you to uncover the
          facts, ensuring honesty and trust prevail in every interaction.
        </div>
        <Button
          component="label"
          role={undefined}
          variant="contained"
          tabIndex={-1}
          startIcon={<CloudUpload />}
          style={{
            backgroundColor: "#ffffff",
            borderRadius: 20,
            padding: "8px 24px",
            color: "#012ca8",
          }}
        >
          Upload file
          <VisuallyHiddenInput type="file" onChange={handleFileChange} />
        </Button>
      </div>
      <div className="hero-right">
        <img
          src="/icons/voice.png"
          style={{ width: 500, visibility: "hidden" }}
        />
      </div>
      {
        <Dialog
          open={open}
          //TransitionComponent={Transition}
          keepMounted
          onClose={toggleDialog}
          aria-describedby="alert-dialog-slide-description"
        >
          <div style={resultStyle}>
            Prediction result: There is a {weightedAverage}% chance that the
            subject in this video is a liar.
          </div>
          <Button
            variant="contained"
            style={{
              width: "30%",
              backgroundColor: "#1d2671",
              borderRadius: 25,
              color: "#fff",
              marginBottom: "25px",
              marginLeft: "200px",
            }}
            onClick={toggleDetailsDialog}
          >
            For more details
          </Button>
        </Dialog>
      }

      {/* New Dialog for More Details */}
      {
        <Dialog
          open={detailsOpen}
          //TransitionComponent={Transition}
          keepMounted
          onClose={toggleDetailsDialog}
          aria-describedby="alert-dialog-slide-description"
        >
          <DialogTitle
            sx={{ color: "#1d2671", textAlign: "center", fontWeight: "bold" }}
          >
            Details
          </DialogTitle>
          <DialogContent>
            <DialogContentText sx={{ fontSize: "20px" }}>
              <div style={{ color: "black" }}>
                {" "}
                Overall prediction: {weightedAverage}% liar.
              </div>
              <div style={{ color: "black" }}>
                Video prediction: {percVideo}% liar.
              </div>
              <div style={{ color: "black" }}>
                Expressions prediction: {percExpression}% liar.
              </div>
              <div style={{ color: "black" }}>
                Audio prediction: {percAudio}% liar.
              </div>
              <div style={{ color: "black" }}>
                Text prediction: {percText}% liar.
              </div>
            </DialogContentText>
          </DialogContent>
          <DialogActions>
            <Button onClick={toggleDetailsDialog} color="primary">
              Close
            </Button>
          </DialogActions>
        </Dialog>
      }
      <Backdrop sx={{ color: "#fff" }} open={isLoading}>
        <CircularProgress color="inherit" />
      </Backdrop>
    </div>
  );
}

export default Hero;
