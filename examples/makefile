# Makefile to automate the Blender-to-dmat pipeline

# Define paths to scripts and tools
CONVERT_DIR=../convert
ANIM_SCRIPT=$(CONVERT_DIR)/export_anim.py
OBJ_SCRIPT=$(CONVERT_DIR)/export_obj.py
TGF_SCRIPT=$(CONVERT_DIR)/export_tgf.py
MSH_TO_MESH_SCRIPT=$(CONVERT_DIR)/msh_to_mesh.py
CREATE_DMAT_SCRIPT=$(CONVERT_DIR)/create_dmat.py
TETWILD_BIN=../fTetWild/build/FloatTetwild_bin

# Entry point target, invoked with: make convert TARGET=blender/target.blend
convert:
	@$(MAKE) pipeline TARGET=$(TARGET)

pipeline:
	@echo "[+] Starting conversion pipeline for $(TARGET)"
	@export NAME=$$(basename $(TARGET) .blend) && \
	mkdir -p $${NAME} && \
	python $(ANIM_SCRIPT) -i $(TARGET) -o $${NAME}/$${NAME}_anim.txt && \
	python $(OBJ_SCRIPT) -i $(TARGET) -o $${NAME}/$${NAME}.obj && \
	python $(TGF_SCRIPT) -i $(TARGET) -o $${NAME}/$${NAME}.tgf && \
	$(TETWILD_BIN) -i $${NAME}/$${NAME}.obj -o $${NAME}/$${NAME} -q --manifold-surface > /dev/null && \
	python $(MSH_TO_MESH_SCRIPT) -i $${NAME}/$${NAME}_.msh -o $${NAME}/$${NAME}.mesh && \
	python $(CREATE_DMAT_SCRIPT) -t $${NAME}/$${NAME}.tgf -m $${NAME}/$${NAME}.mesh -o $${NAME}/$${NAME}.dmat && \
	rm -f $${NAME}/$${NAME}__sf.obj $${NAME}/$${NAME}__tracked_surface.stl $${NAME}/$${NAME}_.csv $${NAME}/$${NAME}_.msh && \
	echo "[+] Conversion pipeline completed for $${NAME}"

.PHONY: convert pipeline